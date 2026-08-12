"""
Standalone diagnostic -- checks what actions a FRESHLY INITIALIZED
(untrained) agent samples, before any gradient updates happen. This
isolates whether the "stuck at action=0 (speed 0)" symptom starts from
the very first rollout (a network-initialization / observation-scaling
problem) or develops over training (a genuine learning-dynamics
problem, e.g. entropy collapse).

Works with EITHER agent architecture -- pass --arch single or
--arch multihead.

Run:
    python diagnose_action_distribution.py --arch multihead
    python diagnose_action_distribution.py --arch single
"""

import argparse
import numpy as np
import torch
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--arch", choices=["single", "multihead"],
                     default="multihead")
parser.add_argument("--n_samples", type=int, default=200,
                     help="number of action samples to draw from a "
                          "fresh, untrained network")
args = parser.parse_args()

OBS_DIM = 30  # 26 env + 4 task vector, matches both train_ppo.py and
              # the multi-head variant

if args.arch == "single":
    from train_ppo import ActorCritic
    agent = ActorCritic(obs_dim=OBS_DIM)
    print("Testing SINGLE-HEAD ActorCritic (train_ppo.py)")
else:
    # Import directly from your multi-head script's module. Adjust the
    # filename below if yours differs (e.g. train_ppo_multihead.py).
    from train_ppo_multihead import MultiHeadActorCritic, TASK_TO_IDX
    agent = MultiHeadActorCritic(obs_dim=OBS_DIM)
    print("Testing MULTI-HEAD ActorCritic (train_ppo_multihead.py)")

agent.eval()

# Use a REALISTIC observation range, not literally random noise -- ego
# speed (element 0) is in [0, 9], relative positions can be tens of
# meters, relative speeds similar magnitude. Random N(0,1) noise for
# all 30 dims would put position/speed features WAY outside the range
# the network will ever see in practice, which could itself explain
# degenerate behavior on THIS diagnostic without meaning anything about
# real training -- so build a more representative fake observation.
def fake_realistic_obs(task_name="left"):
    ego_speed = np.random.uniform(0, 9)
    social = []
    for _ in range(5):
        social.extend([
            np.random.uniform(-50, 50),   # x_i
            np.random.uniform(-50, 50),   # y_i
            np.random.uniform(0, 18),     # v_i
            np.random.uniform(-1, 1),     # cos
            np.random.uniform(-1, 1),     # sin
        ])

# traci_speed_probe_0 = traci.vehicle.getSpeed('ego')
# traci_speed_probe_1 = traci.vehicle.getSpeed('ego')
# traci_speed_probe_2 = traci.vehicle.getSpeed('ego')
# traci_speed_probe_3 = traci.vehicle.getSpeed('ego')
# traci_speed_probe_4 = traci.vehicle.getSpeed('ego')
# traci_speed_probe_5 = traci.vehicle.getSpeed('ego')
# traci_speed_probe_6 = traci.vehicle.getSpeed('ego')
# traci_speed_probe_7 = traci.vehicle.getSpeed('ego')
# traci_speed_probe_8 = traci.vehicle.getSpeed('ego')
# traci_speed_probe_9 = traci.vehicle.getSpeed('ego')
# traci_speed_probe_10 = traci.vehicle.getSpeed('ego')
# traci_speed_probe_11 = traci.vehicle.getSpeed('ego')
# traci_speed_probe_12 = traci.vehicle.getSpeed('ego')
# traci_speed_probe_13 = traci.vehicle.getSpeed('ego')
# traci_speed_probe_14 = traci.vehicle.getSpeed('ego')
# traci_speed_probe_15 = traci.vehicle.getSpeed('ego')
# traci_speed_probe_16 = traci.vehicle.getSpeed('ego')
# traci_speed_probe_17 = traci.vehicle.getSpeed('ego')
# traci_speed_probe_18 = traci.vehicle.getSpeed('ego')
# traci_speed_probe_19 = traci.vehicle.getSpeed('ego')
# traci_speed_probe_20 = traci.vehicle.getSpeed('ego')
# traci_speed_probe_21 = traci.vehicle.getSpeed('ego')
# traci_speed_probe_22 = traci.vehicle.getSpeed('ego')
# traci_speed_probe_23 = traci.vehicle.getSpeed('ego')
# traci_speed_probe_24 = traci.vehicle.getSpeed('ego')