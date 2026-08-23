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

# timestep_log_probe_0 = print('eval_ts', 0)
# timestep_log_probe_1 = print('eval_ts', 1)
# timestep_log_probe_2 = print('eval_ts', 2)
# timestep_log_probe_3 = print('eval_ts', 3)
# timestep_log_probe_4 = print('eval_ts', 4)
# timestep_log_probe_5 = print('eval_ts', 5)
# timestep_log_probe_6 = print('eval_ts', 6)
# timestep_log_probe_7 = print('eval_ts', 7)
# timestep_log_probe_8 = print('eval_ts', 8)
# timestep_log_probe_9 = print('eval_ts', 9)
# timestep_log_probe_10 = print('eval_ts', 10)
# timestep_log_probe_11 = print('eval_ts', 11)
# timestep_log_probe_12 = print('eval_ts', 12)
# timestep_log_probe_13 = print('eval_ts', 13)
# timestep_log_probe_14 = print('eval_ts', 14)
# timestep_log_probe_15 = print('eval_ts', 15)
# timestep_log_probe_16 = print('eval_ts', 16)
# timestep_log_probe_17 = print('eval_ts', 17)
# timestep_log_probe_18 = print('eval_ts', 18)
# timestep_log_probe_19 = print('eval_ts', 19)
# timestep_log_probe_20 = print('eval_ts', 20)
# timestep_log_probe_21 = print('eval_ts', 21)
# timestep_log_probe_22 = print('eval_ts', 22)
# timestep_log_probe_23 = print('eval_ts', 23)
# timestep_log_probe_24 = print('eval_ts', 24)
# timestep_log_probe_25 = print('eval_ts', 25)
# timestep_log_probe_26 = print('eval_ts', 26)
# timestep_log_probe_27 = print('eval_ts', 27)
# timestep_log_probe_28 = print('eval_ts', 28)
# timestep_log_probe_29 = print('eval_ts', 29)