"""
Robust Evaluation Script: Multi-Task DQN vs RLlib PPO Baseline.
Runs 1,000 episodes cycling all 9 scenarios and tasks.
Outputs:
  1. Detailed terminal summary tables.
  2. CSV dataset with per-episode telemetry.
  3. Side-by-side comparison bar charts matching Kai et al. (Fig. 4 & Fig. 5).
"""

import argparse
import csv
import glob
import os
import time
import numpy as np
import torch
from gymnasium import spaces

from flow.envs.base import Env as FlowEnv
from flow.core.params import (TrafficLightParams, NetParams, InitialConfig,
                               VehicleParams, SumoParams, EnvParams, InFlows,
                               SumoCarFollowingParams)
from flow.controllers import IDMController, RLController

from intersection_netw import (UnsignalizedIntersectionNetwork, 
                               SCENARIO_CONFIGS, EGO_ROUTES)
from intersection_env import MultiTaskIntersectionEnv, DISTANCE_BEFORE_STOPLINE
from multitask_dqn_model import MultiTaskDQN, N_ACTIONS

TASKS = ["left", "straight", "right"]
ALL_SCENARIOS = [f"scenario_{c}" for c in "abcdefghi"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPECTED_OBS_DIM = 26

HEAVY_CONGESTION_RATES = {
    "scenario_a": {"north_in": 600,  "south_in": 600,  "east_in": 500},
    "scenario_b": {"north_in": 800,  "south_in": 800,  "east_in": 600},
    "scenario_c": {"north_in": 1000, "south_in": 1000, "east_in": 600},
    "scenario_d": {"north_in": 600,  "south_in": 1100, "east_in": 1000},
    "scenario_e": {"north_in": 1100, "south_in": 600,  "east_in": 1000},
    "scenario_f": {"north_in": 1100, "south_in": 1100, "east_in": 1000},
    "scenario_g": {"north_in": 1300, "south_in": 1300, "east_in": 800},
    "scenario_h": {"north_in": 1500, "south_in": 1000, "east_in": 1000},
    "scenario_i": {"north_in": 1600, "south_in": 1600, "east_in": 1200},
}


def build_stochastic_scenario_env(scenario_name: str):
    veh_params = VehicleParams()
    veh_params.add(
        veh_id="human",
        acceleration_controller=(IDMController, {
            "a": 2.6,
            "b": 4.5,
            "T": 1.0,
            "v0": 18.0,
            "s0": 2.0,
        }),
        car_following_params=SumoCarFollowingParams(
            speed_mode="all_checks",
            min_speed=8.0,
            max_speed=20.0,
            speed_dev=0.2,
            sigma=0.6,
            impatience=0.5,
        ),
        num_vehicles=0,
    )
    veh_params.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        car_following_params=SumoCarFollowingParams(speed_mode="aggressive"),
        num_vehicles=0,
    )

    inflow = InFlows()
    rates = HEAVY_CONGESTION_RATES.get(scenario_name, HEAVY_CONGESTION_RATES["scenario_c"])
    for arm, rate in rates.items():
        inflow.add(
            veh_type="human",
            edge=arm,
            vehs_per_hour=rate,
            depart_speed="random",
            depart_lane="random",
        )

    net_params = NetParams(
        inflows=inflow,
        additional_params=SCENARIO_CONFIGS[scenario_name],
    )

    class _RawFlowEnv(FlowEnv):
        @property
        def action_space(self):
            return spaces.Discrete(N_ACTIONS)

        @property
        def observation_space(self):
            return spaces.Box(low=-np.inf, high=np.inf, shape=(EXPECTED_OBS_DIM,), dtype=np.float32)

        def _apply_rl_actions(self, rl_actions): pass
        def get_state(self): return np.zeros(EXPECTED_OBS_DIM, dtype=np.float32)
        def compute_reward(self, rl_actions, **kwargs): return 0.0

    network = UnsignalizedIntersectionNetwork(
        name=f"eval_robust_{scenario_name}",
        vehicles=veh_params,
        net_params=net_params,
        initial_config=InitialConfig(spacing="random", perturbation=1),
        traffic_lights=TrafficLightParams(),
    )
    raw_env = _RawFlowEnv(
        env_params=EnvParams(horizon=1000, additional_params={"action_set": [0, 3, 6, 9]}, sims_per_step=1),
        sim_params=SumoParams(sim_step=0.1, render=False, restart_instance=True),
        network=network,
    )
    return MultiTaskIntersectionEnv(raw_env)


def resolve_dqn_ckpt(path_or_dir):
    if path_or_dir is None:
        return None
    if os.path.isfile(path_or_dir):
        return path_or_dir
    search_paths = [path_or_dir, os.path.join("checkpoints", path_or_dir)]
    for sp in search_paths:
        if os.path.isdir(sp):
            for candidate in ["ckpt_best.pt", "ckpt_final.pt"]:
                p = os.path.join(sp, candidate)
                if os.path.isfile(p):
                    return p
            pts = sorted(glob.glob(os.path.join(sp, "ckpt_*.pt")))
            if pts:
                return pts[-1]
    raise FileNotFoundError(f"Cannot find DQN checkpoint: {path_or_dir}")


def resolve_rllib_ckpt(path_or_dir):
    if path_or_dir is None:
        return None
    search_paths = [path_or_dir, os.path.join("runs", path_or_dir)]
    for sp in search_paths:
        if os.path.isdir(sp):
            if "policies" in os.listdir(sp) or any(f.startswith("checkpoint") for f in os.listdir(sp)):
                sub = sorted(glob.glob(os.path.join(sp, "checkpoint_*")))
                return sub[-1] if sub else sp
            sub = sorted(glob.glob(os.path.join(sp, "**", "checkpoint_*"), recursive=True))
            if sub:
                return sub[-1]
    raise FileNotFoundError(f"Cannot find RLlib checkpoint: {path_or_dir}")


def load_dqn(ckpt_path):
    model = MultiTaskDQN().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_ppo(ckpt_dir):
    from ray.rllib.policy.policy import Policy
    from ray.rllib.algorithms.algorithm import Algorithm

    try:
        algo = Algorithm.from_checkpoint(ckpt_dir)
        policy = algo.get_policy("default_policy")
    except Exception:
        pol_dir = os.path.join(ckpt_dir, "policies", "default_policy")
        target_dir = pol_dir if os.path.isdir(pol_dir) else ckpt_dir
        policy = Policy.from_checkpoint(target_dir)
    return policy


def dqn_action(model, obs, g):
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        g_t = torch.as_tensor(g, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        R = model(obs_t)
        q = MultiTaskDQN.masked_q(R, g_t)
        return int(torch.argmax(q, dim=1).item())


def ppo_action(policy, obs, g):
    full_obs = np.concatenate([obs, g], dtype=np.float32)
    action, _, _ = policy.compute_single_action(full_obs, explore=False)
    return int(action)


def evaluate_agent(algo_name, action_fn, model, envs_by_scenario,
                   scenarios, total_episodes, max_steps):
    results = []
    num_scenarios = len(scenarios)

    for ep in range(total_episodes):
        sc_name = scenarios[ep % num_scenarios]
        env = envs_by_scenario[sc_name]

        task_mode = TASKS[ep % len(TASKS)]
        obs, _ = env.reset(options={"task": task_mode})
        task_name = env.current_task_name
        g = env.active_g
        ego_id = env._current_ego_id
        target_exit_edge = EGO_ROUTES["west_in"][task_name][-1]

        ep_reward = 0.0
        ep_len = 0
        speeds = []
        success = False
        collided = False
        collision_type = "None"
        last_seen_edge = "west_in"

        for step_i in range(max_steps):
            speeds.append(float(obs[0]))
            action = action_fn(model, obs, g)

            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1
            obs = next_obs

            curr_edge = info.get("edge", None)
            if curr_edge:
                last_seen_edge = curr_edge

            collided_ids = env.flow_env.k.simulation.kernel_api.simulation.getCollidingVehiclesIDList()
            if ego_id in collided_ids or info.get("is_collision", False):
                collided = True
                collision_type = info.get("collision_type", "AT_INTERSECTION")
                break

            if last_seen_edge == target_exit_edge or info.get("is_success", False):
                success = True
                break

            all_ids = env.flow_env.k.vehicle.get_ids()
            if ego_id not in all_ids:
                if last_seen_edge == target_exit_edge or (last_seen_edge and last_seen_edge.startswith(":")):
                    success = True
                break

            if terminated or truncated:
                break

        sim_step = env.flow_env.sim_params.sim_step
        travel_time = ep_len * sim_step if success else None
        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        outcome = "SUCCESS" if success else ("COLLISION (" + collision_type + ")" if collided else "TIMEOUT")

        results.append({
            "algorithm": algo_name,
            "scenario": sc_name,
            "task": task_name,
            "episode": ep + 1,
            "reward": ep_reward,
            "length": ep_len,
            "success": success,
            "collision": collided,
            "collision_type": collision_type,
            "outcome": outcome,
            "travel_time_s": travel_time,
            "avg_speed_mps": avg_speed,
        })

        if (ep + 1) % 50 == 0 or (ep + 1) == total_episodes:
            succ_pct = np.mean([r["success"] for r in results]) * 100
            coll_pct = np.mean([r["collision"] for r in results]) * 100
            valid_tt = [r["travel_time_s"] for r in results if r["travel_time_s"] is not None]
            m_tt = f"{np.mean(valid_tt):.2f}s" if valid_tt else "N/A"
            print(f"  [{algo_name}] Ep {ep+1:4d}/{total_episodes} | "
                  f"Succ: {succ_pct:5.1f}% | Coll: {coll_pct:5.1f}% | AvgTime: {m_tt:>6} | "
                  f"Last: {task_name:8s} -> {outcome}")

    return results


def print_diagnostic_breakdown(results):
    print("\n" + "=" * 90)
    print("COMPARATIVE EVALUATION SUMMARY (OVER 1000 EPISODES)")
    print("=" * 90)
    header = f"{'Task':<10} {'Algorithm':<16} {'Episodes':>8} {'Success':>9} {'Collision':>10} {'Timeout':>9} {'AvgTime(s)':>11} {'MeanSpeed':>11}"
    print(header)
    print("-" * len(header))

    algos = sorted(list(set(r["algorithm"] for r in results)))

    for task in TASKS:
        for algo in algos:
            subset = [r for r in results if r["task"] == task and r["algorithm"] == algo]
            if not subset:
                continue
            succ = np.mean([r["success"] for r in subset]) * 100
            coll = np.mean([r["collision"] for r in subset]) * 100
            tout = 100.0 - (succ + coll)
            tt = [r["travel_time_s"] for r in subset if r["travel_time_s"] is not None]
            avg_t = f"{np.mean(tt):.2f}" if tt else "N/A"
            spds = [r["avg_speed_mps"] for r in subset]
            avg_v = f"{np.mean(spds):.2f} m/s" if spds else "N/A"
            print(f"{task:<10} {algo:<16} {len(subset):8d} {succ:8.1f}% {coll:9.1f}% {tout:8.1f}% {avg_t:>11} {avg_v:>11}")

    print("-" * len(header))
    for algo in algos:
        subset = [r for r in results if r["algorithm"] == algo]
        succ = np.mean([r["success"] for r in subset]) * 100
        coll = np.mean([r["collision"] for r in subset]) * 100
        tout = 100.0 - (succ + coll)
        tt = [r["travel_time_s"] for r in subset if r["travel_time_s"] is not None]
        avg_t = f"{np.mean(tt):.2f}" if tt else "N/A"
        spds = [r["avg_speed_mps"] for r in subset]
        avg_v = f"{np.mean(spds):.2f} m/s" if spds else "N/A"
        print(f"{'OVERALL':<10} {algo:<16} {len(subset):8d} {succ:8.1f}% {coll:9.1f}% {tout:8.1f}% {avg_t:>11} {avg_v:>11}")
    print("=" * 90 + "\n")


def plot_paper_style_results(results, save_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not found; skipping graph generation.")
        return

    os.makedirs(save_dir, exist_ok=True)
    algos = sorted(list(set(r["algorithm"] for r in results)))
    task_display = ["Turning Left", "Going Straight", "Turning Right"]
    task_keys = ["left", "straight", "right"]

    colors = {
        "left": "#7ea6e0",       # Light blue (matching paper)
        "straight": "#f9f871",   # Yellow (matching paper)
        "right": "#f28e8e",      # Coral red (matching paper)
    }

    # -------------------------------------------------------------
    # Figure 1: Success Rate (Horizontal Bar Chart matching Fig. 4)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    y = np.arange(len(algos))
    height = 0.22

    for i, t_key in enumerate(task_keys):
        rates = []
        for algo in algos:
            subset = [r for r in results if r["algorithm"] == algo and r["task"] == t_key]
            val = (np.mean([r["success"] for r in subset]) * 100) if subset else 0.0
            rates.append(val)
        
        offset = (i - 1) * height
        rects = ax.barh(y + offset, rates, height, label=task_display[i], color=colors[t_key], edgecolor="gray")
        
        for rect in rects:
            w = rect.get_width()
            ax.annotate(f"{w:.1f}%",
                        xy=(w, rect.get_y() + rect.get_height() / 2),
                        xytext=(3, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(algos, fontweight="bold")
    ax.set_xlim(0, 115)
    ax.set_xlabel("Success Rate (%)")
    ax.set_title("Fig. 4: Success rate of different algorithms for all tasks (over 1000 episodes)", fontsize=11)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig4_success_rate.png"), dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 2: Average Time (Horizontal Bar Chart matching Fig. 5)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, t_key in enumerate(task_keys):
        times = []
        for algo in algos:
            subset = [r for r in results if r["algorithm"] == algo and r["task"] == t_key]
            tt = [r["travel_time_s"] for r in subset if r["travel_time_s"] is not None]
            times.append(np.mean(tt) if tt else 0.0)

        offset = (i - 1) * height
        rects = ax.barh(y + offset, times, height, label=task_display[i], color=colors[t_key], edgecolor="gray")

        for rect in rects:
            w = rect.get_width()
            ax.annotate(f"{w:.2f}",
                        xy=(w, rect.get_y() + rect.get_height() / 2),
                        xytext=(3, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(algos, fontweight="bold")
    max_t = max([r["travel_time_s"] for r in results if r["travel_time_s"] is not None] or [50.0])
    ax.set_xlim(0, max_t * 1.25)
    ax.set_xlabel("Average Time (s)")
    ax.set_title("Fig. 5: Average time of different algorithms for all tasks (over 1000 episodes)", fontsize=11)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig5_average_time.png"), dpi=300)
    plt.close(fig)

    print(f"[PLOTS SAVED] Fig. 4 and Fig. 5 written to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dqn", type=str, required=True, help="DQN checkpoint folder or .pt")
    parser.add_argument("--ppo", type=str, required=True, help="RLlib PPO experiment directory")
    parser.add_argument("--episodes", type=int, default=1000, help="Total episodes per algorithm")
    parser.add_argument("--max_steps", type=int, default=800, help="Max steps per episode")
    args = parser.parse_args()

    dqn_path = resolve_dqn_ckpt(args.dqn)
    ppo_path = resolve_rllib_ckpt(args.ppo)

    print(f"[INIT] Pre-building simulation environments across all 9 scenarios...")
    envs = {sc: build_stochastic_scenario_env(sc) for sc in ALL_SCENARIOS}

    all_results = []
    try:
        print(f"\n[RUNNING] Multi-Task DQN Evaluation ({os.path.basename(dqn_path)}) over {args.episodes} episodes...")
        dqn_net = load_dqn(dqn_path)
        all_results.extend(evaluate_agent(
            "Multi-Task DQN", dqn_action, dqn_net, envs,
            ALL_SCENARIOS, args.episodes, args.max_steps
        ))

        print(f"\n[RUNNING] RLlib PPO Evaluation ({os.path.basename(ppo_path)}) over {args.episodes} episodes...")
        ppo_pol = load_ppo(ppo_path)
        all_results.extend(evaluate_agent(
            "PPO", ppo_action, ppo_pol, envs,
            ALL_SCENARIOS, args.episodes, args.max_steps
        ))
    finally:
        for sc_env in envs.values():
            try:
                sc_env.flow_env.terminate()
            except Exception:
                pass

    print_diagnostic_breakdown(all_results)

    out_dir = os.path.join("eval_results", time.strftime("eval_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    csv_file = os.path.join(out_dir, "evaluation_data_1000ep.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

    plot_paper_style_results(all_results, out_dir)
# perturbation_inflow_test_0 = {'delta': 0}
# perturbation_inflow_test_1 = {'delta': 2}
# perturbation_inflow_test_2 = {'delta': 4}
# perturbation_inflow_test_3 = {'delta': 6}
# perturbation_inflow_test_4 = {'delta': 8}
# perturbation_inflow_test_5 = {'delta': 10}
# perturbation_inflow_test_6 = {'delta': 12}
# perturbation_inflow_test_7 = {'delta': 14}
# perturbation_inflow_test_8 = {'delta': 16}
# perturbation_inflow_test_9 = {'delta': 18}
# perturbation_inflow_test_10 = {'delta': 20}
# perturbation_inflow_test_11 = {'delta': 22}
# perturbation_inflow_test_12 = {'delta': 24}
# perturbation_inflow_test_13 = {'delta': 26}
# perturbation_inflow_test_14 = {'delta': 28}
# perturbation_inflow_test_15 = {'delta': 30}
# perturbation_inflow_test_16 = {'delta': 32}
# perturbation_inflow_test_17 = {'delta': 34}
# perturbation_inflow_test_18 = {'delta': 36}
# perturbation_inflow_test_19 = {'delta': 38}
# perturbation_inflow_test_20 = {'delta': 40}
# perturbation_inflow_test_21 = {'delta': 42}
# perturbation_inflow_test_22 = {'delta': 44}
# perturbation_inflow_test_23 = {'delta': 46}
# perturbation_inflow_test_24 = {'delta': 48}