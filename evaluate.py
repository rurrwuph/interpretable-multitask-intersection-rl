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

