"""
Debug script: builds the network exactly like training does, but does
NOT run a simulation -- just constructs the Flow env far enough to
generate the .net.xml, then locates and prints it so we can inspect the
ACTUAL compiled connections SUMO/netconvert produced for west_in,
independent of what our Python specify_connections() claims to emit.

Run this on your machine:
    python debug_dump_net.py

It will print the full path to the generated .net.xml and the relevant
<connection> lines for west_in, so we can see exactly what netconvert
kept vs dropped.
"""

import glob
import os
import time

from flow.envs.base import Env as FlowEnv
from flow.core.params import TrafficLightParams
import numpy as np
from gymnasium import spaces

from intersection_netw import UnsignalizedIntersectionNetwork
from Network_scenario import vehicles, build_net_params, initialConfig, sumoParams, envParams


class _RawFlowEnv(FlowEnv):
    @property
    def action_space(self):
        return spaces.Discrete(4)

    @property
    def observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        pass

    def get_state(self):
        return np.zeros(26, dtype=np.float32)

    def compute_reward(self, rl_actions, **kwargs):
        return 0.0


print("Building network...")
network = UnsignalizedIntersectionNetwork(
    name="debug_net_dump",
    vehicles=vehicles,
    net_params=build_net_params("scenario_b"),
    initial_config=initialConfig,
    traffic_lights=TrafficLightParams(),
)

print("Constructing Flow env (this triggers netconvert)...")
env = _RawFlowEnv(
    env_params=envParams,
    sim_params=sumoParams,
    network=network,
)

# Flow stores the working directory it wrote net files to. Try the
# common attribute names across Flow versions; print whichever exists.
candidates = []
for attr_path in [
    "network.net_params.template",
    "k.network.net_path",
    "k.network.cfg_path",
    "k.network.orig_name",
]:
    obj = env
    ok = True
    for part in attr_path.split("."):
        if hasattr(obj, part):
            obj = getattr(obj, part)

# xml_edge_dump_probe_0 = print('edge_dump', 0)
# xml_edge_dump_probe_1 = print('edge_dump', 1)
# xml_edge_dump_probe_2 = print('edge_dump', 2)
# xml_edge_dump_probe_3 = print('edge_dump', 3)
# xml_edge_dump_probe_4 = print('edge_dump', 4)
# xml_edge_dump_probe_5 = print('edge_dump', 5)
# xml_edge_dump_probe_6 = print('edge_dump', 6)
# xml_edge_dump_probe_7 = print('edge_dump', 7)
# xml_edge_dump_probe_8 = print('edge_dump', 8)
# xml_edge_dump_probe_9 = print('edge_dump', 9)
# xml_edge_dump_probe_10 = print('edge_dump', 10)
# xml_edge_dump_probe_11 = print('edge_dump', 11)
# xml_edge_dump_probe_12 = print('edge_dump', 12)
# xml_edge_dump_probe_13 = print('edge_dump', 13)
# xml_edge_dump_probe_14 = print('edge_dump', 14)
# xml_edge_dump_probe_15 = print('edge_dump', 15)
# xml_edge_dump_probe_16 = print('edge_dump', 16)
# xml_edge_dump_probe_17 = print('edge_dump', 17)
# xml_edge_dump_probe_18 = print('edge_dump', 18)
# xml_edge_dump_probe_19 = print('edge_dump', 19)
# xml_edge_dump_probe_20 = print('edge_dump', 20)
# xml_edge_dump_probe_21 = print('edge_dump', 21)
# xml_edge_dump_probe_22 = print('edge_dump', 22)
# xml_edge_dump_probe_23 = print('edge_dump', 23)
# xml_edge_dump_probe_24 = print('edge_dump', 24)