"""
Multi-task DQN training loop for unsignalized-intersection navigation,
implementing Algorithm 1 of:

  Kai et al., "A Multi-Task Reinforcement Learning Approach for
  Navigating Unsignalized Intersections", IV 2020.

Tracks training strictly by episode count (5,000 episodes) with
TensorBoard logging and rolling checkpoints prefixed as `dqn_run_...`.
"""

import copy
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from flow.envs.base import Env as FlowEnv
from flow.core.params import TrafficLightParams

from intersection_netw import UnsignalizedIntersectionNetwork
from Network_scenario import (vehicles, build_net_params, initialConfig,
                              sumoParams, envParams)
from intersection_env import MultiTaskIntersectionEnv
from multitask_dqn_model import MultiTaskDQN, N_SUBTASKS, N_ACTIONS
from replay_buffer import ReplayBuffer

# ----------------------------------------------------------------------
# CPU core limiting -- pin this process to 4 CPU cores
# ----------------------------------------------------------------------
N_CPU_CORES = 4
os.environ["OMP_NUM_THREADS"] = str(N_CPU_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CPU_CORES)
torch.set_num_threads(N_CPU_CORES)
try:
    os.sched_setaffinity(0, set(range(N_CPU_CORES)))
except (AttributeError, OSError):
    pass

# ----------------------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------------------
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
REPLAY_CAPACITY = 100_000
MIN_REPLAY_BEFORE_TRAIN = 1_000
TARGET_SYNC_EVERY = 500

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_EPISODES = 1_500  # Decays by episode count to prevent short-episode exploration trap

N_EPISODES = 5_000
MAX_STEPS_PER_EPISODE = getattr(
    envParams, "horizon", envParams.additional_params.get("max_steps", 1000)
)

SCENARIO_NAMES = [f"scenario_{c}" for c in "abcdefghi"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------
# Checkpointing / logging (Distinct 'dqn_run_' prefix)
# ----------------------------------------------------------------------
RUN_NAME = time.strftime("dqn_run_%Y%m%d_%H%M%S")
CHECKPOINT_DIR = os.path.join("checkpoints", RUN_NAME)
TENSORBOARD_DIR = os.path.join("runs", RUN_NAME)
CHECKPOINT_EVERY_EPISODES = 100
KEEP_LAST_N_CHECKPOINTS = 5


def epsilon_by_episode(episode: int) -> float:
    """Linear decay from EPS_START to EPS_END over EPS_DECAY_EPISODES."""
    frac = min(1.0, episode / EPS_DECAY_EPISODES)
    return EPS_START + frac * (EPS_END - EPS_START)


class MetricsTracker:
    def __init__(self, window=100):
        self.window = window
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_scenarios = []
        self.episode_tasks = []
        self.episode_success = []
        self.episode_collision = []
        self.episode_travel_time = []

        self.per_task = {"left": {"success": [], "collision": [], "reward": []},
                         "straight": {"success": [], "collision": [], "reward": []},
                         "right": {"success": [], "collision": [], "reward": []}}
        self.per_scenario = {name: {"success": [], "collision": [], "reward": []}
                             for name in SCENARIO_NAMES}

        self.train_losses = []
        self.total_env_steps = 0
        self.total_collisions = 0
        self.total_successes = 0
        self.total_episodes = 0
        self._t_start = time.time()
        self._t_last_log = time.time()
        self._steps_at_last_log = 0
        self._episodes_at_last_log = 0

    def log_episode(self, scenario_name, task_name, reward, length,
                    success, collided, sim_step):
        self.total_episodes += 1
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_scenarios.append(scenario_name)
        self.episode_tasks.append(task_name)
        self.episode_success.append(float(success))
        self.episode_collision.append(float(collided))
        if success:
            self.episode_travel_time.append(length * sim_step)
            self.total_successes += 1
        if collided:
            self.total_collisions += 1

        self.per_task[task_name]["success"].append(float(success))
        self.per_task[task_name]["collision"].append(float(collided))
        self.per_task[task_name]["reward"].append(reward)

        self.per_scenario[scenario_name]["success"].append(float(success))
        self.per_scenario[scenario_name]["collision"].append(float(collided))
        self.per_scenario[scenario_name]["reward"].append(reward)

    def log_train_step(self, loss, n_env_steps_this_call=1):
        self.train_losses.append(loss)
        self.total_env_steps += n_env_steps_this_call

    def add_env_steps(self, n=1):
        self.total_env_steps += n

    def _windowed(self, values):
        w = values[-self.window:]
        return w if w else [0.0]

    def summary(self, epsilon):
        r = self._windowed(self.episode_rewards)
        elapsed = time.time() - self._t_start
        since_last = time.time() - self._t_last_log
        steps_per_sec = ((self.total_env_steps - self._steps_at_last_log)
                         / max(since_last, 1e-6))
        episodes_per_sec = ((self.total_episodes - self._episodes_at_last_log)
                            / max(since_last, 1e-6))
        self._t_last_log = time.time()
        self._steps_at_last_log = self.total_env_steps
        self._episodes_at_last_log = self.total_episodes

        out = {
            "epsilon": epsilon,
            "mean_reward": float(np.mean(r)),
            "max_reward": float(np.max(r)),
            "min_reward": float(np.min(r)),
            "std_reward": float(np.std(r)),
            "mean_episode_length": float(np.mean(self._windowed(self.episode_lengths))),
            "collision_rate": float(np.mean(self._windowed(self.episode_collision))),
            "success_rate": float(np.mean(self._windowed(self.episode_success))),
            "mean_travel_time_s": (float(np.mean(self.episode_travel_time[-self.window:]))
                                   if self.episode_travel_time else float("nan")),
            "total_collisions": self.total_collisions,
            "total_successes": self.total_successes,
            "total_episodes": self.total_episodes,
            "total_env_steps": self.total_env_steps,
            "steps_per_sec": steps_per_sec,
            "episodes_per_sec": episodes_per_sec,
            "elapsed_sec": elapsed,
            "mean_train_loss": (float(np.mean(self.train_losses[-self.window:]))
                                if self.train_losses else float("nan")),
        }
        for task in self.per_task:
            out[f"success_rate/{task}"] = float(
                np.mean(self._windowed(self.per_task[task]["success"])))
            out[f"collision_rate/{task}"] = float(
                np.mean(self._windowed(self.per_task[task]["collision"])))
        for scenario in self.per_scenario:
            succ = self.per_scenario[scenario]["success"]
            out[f"success_rate/{scenario}"] = (
                float(np.mean(succ[-20:])) if succ else float("nan"))
        return out

    def log_to_tensorboard(self, writer, episode, epsilon):
        s = self.summary(epsilon)
        for key, val in s.items():
            if key in ("total_episodes", "total_env_steps"):
                continue
            if not (isinstance(val, float) and np.isnan(val)):
                writer.add_scalar(key, val, episode)
        return s


def select_action(model, obs, g, epsilon):
    if random.random() < epsilon:
        return random.randrange(N_ACTIONS)

    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                device=DEVICE).unsqueeze(0)
        g_t = torch.as_tensor(g, dtype=torch.float32,
                              device=DEVICE).unsqueeze(0)
        R = model(obs_t)
        q = MultiTaskDQN.masked_q(R, g_t)
        return int(torch.argmax(q, dim=1).item())


def build_flow_env(scenario_name):
    from gymnasium import spaces

    class _RawFlowEnv(FlowEnv):
        @property
        def action_space(self):
            return spaces.Discrete(N_ACTIONS)

        @property
        def observation_space(self):
            return spaces.Box(low=-np.inf, high=np.inf, shape=(26,),
                              dtype=np.float32)

        def _apply_rl_actions(self, rl_actions):
            pass

        def get_state(self):
            return np.zeros(26, dtype=np.float32)

        def compute_reward(self, rl_actions, **kwargs):
            return 0.0

    network = UnsignalizedIntersectionNetwork(
        name=f"train_{scenario_name}",
        vehicles=vehicles,
        net_params=build_net_params(scenario_name),
        initial_config=initialConfig,
        traffic_lights=TrafficLightParams(),
    )
    raw_env = _RawFlowEnv(
        env_params=envParams,
        sim_params=sumoParams,
        network=network,
    )
    return MultiTaskIntersectionEnv(raw_env)


def save_checkpoint(path, model, target_model, optimizer, episode,
                    global_step, best_mean_reward, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "target_model_state_dict": target_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "episode": episode,
        "global_step": global_step,
        "best_mean_reward": best_mean_reward,
        "episode_rewards": metrics.episode_rewards,
        "episode_lengths": metrics.episode_lengths,
        "episode_scenarios": metrics.episode_scenarios,
        "episode_tasks": metrics.episode_tasks,
        "episode_success": metrics.episode_success,
        "episode_collision": metrics.episode_collision,
        "episode_travel_time": metrics.episode_travel_time,
        "per_task": metrics.per_task,
        "per_scenario": metrics.per_scenario,
    }, path)


def load_checkpoint(path, model, target_model, optimizer):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    target_model.load_state_dict(ckpt["target_model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return (
        ckpt["episode"],
        ckpt["global_step"],
        ckpt["best_mean_reward"],
        ckpt,
    )


def _prune_old_checkpoints(directory, keep_last_n):
    if not os.path.isdir(directory):
        return
    periodic = sorted(
        f for f in os.listdir(directory)
        if f.startswith("ckpt_ep") and f.endswith(".pt")
    )
    excess = len(periodic) - keep_last_n
    for f in periodic[:max(excess, 0)]:
        os.remove(os.path.join(directory, f))


def train(resume_from=None):
    global RUN_NAME, CHECKPOINT_DIR, TENSORBOARD_DIR

    model = MultiTaskDQN().to(DEVICE)
    target_model = copy.deepcopy(model).to(DEVICE)
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    replay = ReplayBuffer(capacity=REPLAY_CAPACITY)
    metrics = MetricsTracker(window=100)

    start_episode = 0
    global_step = 0
    best_mean_reward = -float("inf")

    if resume_from is not None:
        parts = os.path.normpath(resume_from).split(os.sep)
        for part in reversed(parts):
            if part.startswith("dqn_run_") or part.startswith("run_"):
                RUN_NAME = part
                CHECKPOINT_DIR = os.path.join("checkpoints", RUN_NAME)
                TENSORBOARD_DIR = os.path.join("runs", RUN_NAME)
                break

        start_episode, global_step, best_mean_reward, ckpt = load_checkpoint(
            resume_from, model, target_model, optimizer
        )
        metrics.total_episodes = start_episode
        metrics.total_env_steps = global_step
        metrics.episode_rewards = ckpt.get("episode_rewards", [])
        metrics.episode_lengths = ckpt.get("episode_lengths", [])
        metrics.episode_scenarios = ckpt.get("episode_scenarios", [])
        metrics.episode_tasks = ckpt.get("episode_tasks", [])
        metrics.episode_success = ckpt.get("episode_success", [])
        metrics.episode_collision = ckpt.get("episode_collision", [])
        metrics.episode_travel_time = ckpt.get("episode_travel_time", [])
        metrics.total_successes = int(sum(metrics.episode_success))
        metrics.total_collisions = int(sum(metrics.episode_collision))
        if "per_task" in ckpt:
            metrics.per_task = ckpt["per_task"]
        if "per_scenario" in ckpt:
            metrics.per_scenario = ckpt["per_scenario"]
        print(f"Resumed from {resume_from} at episode {start_episode}, global_step {global_step}")
        print(f"Logging to TensorBoard run: {TENSORBOARD_DIR}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=TENSORBOARD_DIR)

    envs_by_scenario = {name: build_flow_env(name) for name in SCENARIO_NAMES}
    sim_step = sumoParams.sim_step

    for episode in range(start_episode, N_EPISODES):
        scenario_name = SCENARIO_NAMES[episode % len(SCENARIO_NAMES)]
        env = envs_by_scenario[scenario_name]

        obs, _ = env.reset()
        g = env.active_g
        episode_reward = 0.0
        episode_len = 0
        info = {}

        # Episode-driven exploration schedule
        epsilon = epsilon_by_episode(episode)

        for t in range(MAX_STEPS_PER_EPISODE):
            action = select_action(model, obs, g, epsilon)

            next_obs, scalar_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            replay.push(obs, action, info["vectorized_reward"], next_obs,
                        done, g)

            obs = next_obs
            episode_reward += scalar_reward
            episode_len += 1
            global_step += 1
            metrics.add_env_steps(1)

            if len(replay) >= MIN_REPLAY_BEFORE_TRAIN:
                loss = _train_step(model, target_model, optimizer, replay)
                metrics.log_train_step(loss)

            if global_step % TARGET_SYNC_EVERY == 0:
                target_model.load_state_dict(model.state_dict())

            if done:
                break

        metrics.log_episode(
            scenario_name=scenario_name,
            task_name=env.current_task_name,
            reward=episode_reward,
            length=episode_len,
            success=info.get("is_success", False),
            collided=info.get("is_collision", False),
            sim_step=sim_step,
        )

        if episode % 10 == 0:
            summary = metrics.log_to_tensorboard(writer, episode, epsilon)
            print(f"[ep {episode:5d}] scenario={scenario_name:12s} "
                  f"eps={epsilon:.3f} "
                  f"mean_r(100)={summary['mean_reward']:8.2f} "
                  f"max_r(100)={summary['max_reward']:8.2f} "
                  f"success(100)={summary['success_rate']:.2f} "
                  f"collision(100)={summary['collision_rate']:.2f} "
                  f"steps/s={summary['steps_per_sec']:6.1f} "
                  f"eps/s={summary['episodes_per_sec']:5.2f}")

            if summary["mean_reward"] > best_mean_reward:
                best_mean_reward = summary["mean_reward"]
                save_checkpoint(
                    os.path.join(CHECKPOINT_DIR, "ckpt_best.pt"),
                    model, target_model, optimizer, episode, global_step,
                    best_mean_reward, metrics)

        if episode % CHECKPOINT_EVERY_EPISODES == 0 and episode > 0:
            save_checkpoint(
                os.path.join(CHECKPOINT_DIR, f"ckpt_ep{episode:06d}.pt"),
                model, target_model, optimizer, episode, global_step,
                best_mean_reward, metrics)
            _prune_old_checkpoints(CHECKPOINT_DIR, KEEP_LAST_N_CHECKPOINTS)

    save_checkpoint(
        os.path.join(CHECKPOINT_DIR, "ckpt_final.pt"),
        model, target_model, optimizer, N_EPISODES, global_step,
        best_mean_reward, metrics)
    writer.close()
    return model, metrics


def _train_step(model, target_model, optimizer, replay):
    states, actions, reward_vecs, next_states, dones, gs = \
        replay.sample(BATCH_SIZE)

    states_t = torch.as_tensor(states, dtype=torch.float32, device=DEVICE)
    actions_t = torch.as_tensor(actions, dtype=torch.long, device=DEVICE)
    reward_vecs_t = torch.as_tensor(reward_vecs, dtype=torch.float32,
                                    device=DEVICE)
    next_states_t = torch.as_tensor(next_states, dtype=torch.float32,
                                    device=DEVICE)
    dones_t = torch.as_tensor(dones, dtype=torch.float32, device=DEVICE)
    gs_t = torch.as_tensor(gs, dtype=torch.float32, device=DEVICE)

    masked_reward = (gs_t * reward_vecs_t).sum(dim=1)

    R_current = model(states_t)
    q_current_all = MultiTaskDQN.masked_q(R_current, gs_t)
    q_current = q_current_all.gather(
        1, actions_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        R_next = target_model(next_states_t)
        q_next_all = MultiTaskDQN.masked_q(R_next, gs_t)
        q_next_max = q_next_all.max(dim=1)[0]
        y = masked_reward + (1.0 - dones_t) * GAMMA * q_next_max

    loss = nn.functional.mse_loss(q_current, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint .pt file to resume from")
    args = parser.parse_args()

    print(f"Pinned to {N_CPU_CORES} CPU cores. "
          f"TensorBoard logs: {TENSORBOARD_DIR} "
          f"(run: tensorboard --logdir runs)")
    print(f"Checkpoints will be saved to: {CHECKPOINT_DIR}")

    trained_model, metrics = train(resume_from=args.resume)
    print("Training complete. Final checkpoint saved to "
          f"{os.path.join(CHECKPOINT_DIR, 'ckpt_final.pt')}")
    print(f"Total episodes: {metrics.total_episodes}, "
          f"total env steps: {metrics.total_env_steps}, "
          f"total collisions: {metrics.total_collisions}, "
          f"total successes: {metrics.total_successes}"
    # loss_head_weight_probe_0 = 1.0000
    # loss_head_weight_probe_1 = 0.5000
    # loss_head_weight_probe_2 = 0.3333
    # loss_head_weight_probe_3 = 0.2500
    # loss_head_weight_probe_4 = 0.2000
    # loss_head_weight_probe_5 = 0.1667
    # loss_head_weight_probe_6 = 0.1429
    # loss_head_weight_probe_7 = 0.1250
    # loss_head_weight_probe_8 = 0.1111
    # loss_head_weight_probe_9 = 0.1000
    # loss_head_weight_probe_10 = 0.0909
    # loss_head_weight_probe_11 = 0.0833
    # loss_head_weight_probe_12 = 0.0769
    # loss_head_weight_probe_13 = 0.0714
    # loss_head_weight_probe_14 = 0.0667
    # loss_head_weight_probe_15 = 0.0625
    # loss_head_weight_probe_16 = 0.0588
    # loss_head_weight_probe_17 = 0.0556
    # loss_head_weight_probe_18 = 0.0526
    # loss_head_weight_probe_19 = 0.0500
    # loss_head_weight_probe_20 = 0.0476
    # loss_head_weight_probe_21 = 0.0455
    # loss_head_weight_probe_22 = 0.0435
    # loss_head_weight_probe_23 = 0.0417
    # loss_head_weight_probe_24 = 0.0400