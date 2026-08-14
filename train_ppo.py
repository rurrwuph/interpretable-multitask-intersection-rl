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
