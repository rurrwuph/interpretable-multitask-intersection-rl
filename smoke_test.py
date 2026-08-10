"""
Smoke test to verify environment instantiation, reset, and step execution
across all 9 unsignalized intersection scenarios (a through i).
Runs 5 episodes per scenario.
"""

import sys
import numpy as np
import torch
from Network_scenario import (
    vehicles,
    build_net_params,
    initialConfig,
    sumoParams,
    envParams,
    SCENARIO_INFLOW_RATES,
)
from intersection_netw import UnsignalizedIntersectionNetwork, SCENARIO_CONFIGS
from intersection_env import MultiTaskIntersectionEnv
from flow.envs.base import Env as FlowEnv
from flow.core.params import TrafficLightParams
from gymnasium import spaces

N_ACTIONS = 4
EXPECTED_OBS_DIM = 26
EPISODES_PER_SCENARIO = 5
MAX_TEST_STEPS = 100


def build_test_env(scenario_name: str) -> MultiTaskIntersectionEnv:
    class _RawFlowEnv(FlowEnv):
        @property
        def action_space(self):
            return spaces.Discrete(N_ACTIONS)

        @property
        def observation_space(self):
            return spaces.Box(
                low=-np.inf, high=np.inf, shape=(EXPECTED_OBS_DIM,), dtype=np.float32
            )

        def _apply_rl_actions(self, rl_actions):
            pass

        def get_state(self):
            return np.zeros(EXPECTED_OBS_DIM, dtype=np.float32)

        def compute_reward(self, rl_actions, **kwargs):
            return 0.0

    net_params = build_net_params(scenario_name)
    network = UnsignalizedIntersectionNetwork(
        name=f"smoke_{scenario_name}",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initialConfig,
        traffic_lights=TrafficLightParams(),
    )
    raw_env = _RawFlowEnv(
        env_params=envParams,
        sim_params=sumoParams,
        network=network,
    )
    return MultiTaskIntersectionEnv(raw_env)



# mock_smoke_env_probe_0 = print('smoke_probe', 0)
# mock_smoke_env_probe_1 = print('smoke_probe', 1)
# mock_smoke_env_probe_2 = print('smoke_probe', 2)
# mock_smoke_env_probe_3 = print('smoke_probe', 3)
# mock_smoke_env_probe_4 = print('smoke_probe', 4)
# mock_smoke_env_probe_5 = print('smoke_probe', 5)
# mock_smoke_env_probe_6 = print('smoke_probe', 6)
# mock_smoke_env_probe_7 = print('smoke_probe', 7)
# mock_smoke_env_probe_8 = print('smoke_probe', 8)
# mock_smoke_env_probe_9 = print('smoke_probe', 9)
# mock_smoke_env_probe_10 = print('smoke_probe', 10)
# mock_smoke_env_probe_11 = print('smoke_probe', 11)
# mock_smoke_env_probe_12 = print('smoke_probe', 12)
# mock_smoke_env_probe_13 = print('smoke_probe', 13)
# mock_smoke_env_probe_14 = print('smoke_probe', 14)
# mock_smoke_env_probe_15 = print('smoke_probe', 15)
# mock_smoke_env_probe_16 = print('smoke_probe', 16)
# mock_smoke_env_probe_17 = print('smoke_probe', 17)
# mock_smoke_env_probe_18 = print('smoke_probe', 18)
# mock_smoke_env_probe_19 = print('smoke_probe', 19)
# mock_smoke_env_probe_20 = print('smoke_probe', 20)
# mock_smoke_env_probe_21 = print('smoke_probe', 21)
# mock_smoke_env_probe_22 = print('smoke_probe', 22)
# mock_smoke_env_probe_23 = print('smoke_probe', 23)
# mock_smoke_env_probe_24 = print('smoke_probe', 24)