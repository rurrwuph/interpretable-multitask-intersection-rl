"""
Ray RLlib PPO Baseline for Multi-Task Unsignalized Intersection Navigation.
Implements standard PPO via RLlib with multi-task conditioning and custom
metrics logging (success rate, collisions, travel time, per-task metrics).
"""

import os
import time
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from flow.envs.base import Env as FlowEnv
from flow.core.params import TrafficLightParams

from intersection_netw import UnsignalizedIntersectionNetwork
from Network_scenario import (vehicles, build_net_params, initialConfig,
                              sumoParams, envParams)
from intersection_env import MultiTaskIntersectionEnv

# ----------------------------------------------------------------------
# Pin CPU threads
# ----------------------------------------------------------------------
N_CPU_CORES = 4
os.environ["OMP_NUM_THREADS"] = str(N_CPU_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CPU_CORES)
try:
    os.sched_setaffinity(0, set(range(N_CPU_CORES)))
except (AttributeError, OSError):
    pass

SCENARIO_NAMES = [f"scenario_{c}" for c in "abcdefghi"]
ENV_OBS_DIM = 26
TASK_VEC_DIM = 4
OBS_DIM = ENV_OBS_DIM + TASK_VEC_DIM  # 30
N_ACTIONS = 4


# ----------------------------------------------------------------------
# RLlib Compatible Multi-Task Environment Wrapper
# ----------------------------------------------------------------------
class RLlibIntersectionWrapper(gym.Env):
    """Wraps MultiTaskIntersectionEnv into a 30-dim observation space

    and cycles scenarios across resets for RLlib rollouts.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.scenarios = self.config.get("scenarios", SCENARIO_NAMES)
        self.episode_counter = 0

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        self._envs = {}
        for sc in self.scenarios:
            self._envs[sc] = self._build_underlying_env(sc)
        
        self._current_sc = self.scenarios[0]
        self._current_env = self._envs[self._current_sc]

    def _build_underlying_env(self, scenario_name):
        class _RawFlowEnv(FlowEnv):
            @property
            def action_space(self):
                return spaces.Discrete(N_ACTIONS)

            @property
            def observation_space(self):
                return spaces.Box(low=-np.inf, high=np.inf, shape=(ENV_OBS_DIM,), dtype=np.float32)

            def _apply_rl_actions(self, rl_actions): pass
            def get_state(self): return np.zeros(ENV_OBS_DIM, dtype=np.float32)
            def compute_reward(self, rl_actions, **kwargs): return 0.0

        network = UnsignalizedIntersectionNetwork(
            name=f"rllib_{scenario_name}",
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

    def reset(self, *, seed=None, options=None):
        self._current_sc = self.scenarios[self.episode_counter % len(self.scenarios)]
        self._current_env = self._envs[self._current_sc]
        self.episode_counter += 1

        raw_obs, info = self._current_env.reset(seed=seed, options=options)
        full_obs = np.concatenate([raw_obs, self._current_env.active_g], dtype=np.float32)
        return full_obs, info

    def step(self, action):
        next_raw_obs, reward, terminated, truncated, info = self._current_env.step(action)
        full_obs = np.concatenate([next_raw_obs, self._current_env.active_g], dtype=np.float32)
        
        # Add metadata for RLlib metrics callback
        info["scenario"] = self._current_sc
        info["task_name"] = self._current_env.current_task_name
        return full_obs, reward, terminated, truncated, info


# ----------------------------------------------------------------------
# Custom Metrics Callback for RLlib
# ----------------------------------------------------------------------
class CustomMetricsCallback(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index=0, **kwargs):
        last_info = episode.last_info_for()
        if not last_info:
            return

        is_success = float(last_info.get("is_success", False))
        is_collision = float(last_info.get("is_collision", False))
        task_name = last_info.get("task_name", "unknown")
        scenario = last_info.get("scenario", "unknown")
        sim_step = sumoParams.sim_step

        episode.custom_metrics["success_rate"] = is_success
        episode.custom_metrics["collision_rate"] = is_collision
        episode.custom_metrics[f"success_{task_name}"] = is_success
        episode.custom_metrics[f"collision_{task_name}"] = is_collision
        episode.custom_metrics[f"success_{scenario}"] = is_success

        if is_success:
            episode.custom_metrics["travel_time_s"] = episode.length * sim_step


# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000, help="Total episodes to train")
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True, num_cpus=N_CPU_CORES)
    tune.register_env("MultiTaskIntersectionRLlib-v0", lambda cfg: RLlibIntersectionWrapper(cfg))

    run_name = time.strftime("ppo_rllib_%Y%m%d_%H%M%S")
    local_dir = os.path.abspath("./runs")

    config = (
        PPOConfig()
        .environment(
            env="MultiTaskIntersectionRLlib-v0",
            env_config={"scenarios": SCENARIO_NAMES},
        )
        .framework("torch")
        # Keep num_rollout_workers=0 to run on driver process, preventing SUMO TraCI port collisions
        .rollouts(
            num_rollout_workers=0,
            rollout_fragment_length="auto",
        )
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.05,
            vf_loss_coeff=0.5,
            grad_clip=0.5,
            train_batch_size=2048,
            sgd_minibatch_size=64,
            num_sgd_iter=4,
            model={
                "fcnet_hiddens": [128, 128],
                "fcnet_activation": "tanh",
            },
        )
        .callbacks(CustomMetricsCallback)
        .resources(num_gpus=0)
    )

    tune.run(
        "PPO",
        name=run_name,
        config=config.to_dict(),
        stop={"episodes_total": args.episodes},
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir=local_dir,
    )
# entropy bonus coefficient adjusted to 0.05
