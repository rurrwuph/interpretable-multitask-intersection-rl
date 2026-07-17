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
    env_obs = np.array([ego_speed] + social, dtype=np.float32)
    task_g = {
        "left": [1, 0, 0, 1], "straight": [0, 1, 0, 1], "right": [0, 0, 1, 1]
    }[task_name]
    return np.concatenate([env_obs, task_g], dtype=np.float32)


action_counts = Counter()
speed_map = {0: 0.0, 1: 3.0, 2: 6.0, 3: 9.0}

for i in range(args.n_samples):
    task_name = ["left", "straight", "right"][i % 3]
    obs = fake_realistic_obs(task_name)
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        if args.arch == "single":
            action, logprob, entropy, value = agent.get_action_and_value(obs_t)
        else:
            task_idx = TASK_TO_IDX[task_name]
            task_t = torch.tensor([task_idx], dtype=torch.long)
            action, logprob, entropy, value = agent.get_action_and_value(
                obs_t, task_t)

    action_counts[action.item()] += 1

print()
print(f"Action distribution over {args.n_samples} samples from a FRESH, "
      f"UNTRAINED network (realistic-range observations, tasks cycled "
      f"left/straight/right):")
for a in sorted(action_counts):
    speed = speed_map[a]
    pct = 100 * action_counts[a] / args.n_samples
    bar = "#" * int(pct / 2)
    print(f"  action={a} (speed={speed:4.1f} m/s): {action_counts[a]:4d} "
          f"({pct:5.1f}%) {bar}")

print()
zero_pct = 100 * action_counts.get(0, 0) / args.n_samples
if zero_pct > 60:
    print(f"DIAGNOSIS: action=0 (speed 0) dominates ({zero_pct:.1f}%) even "
          f"in a FRESH, UNTRAINED network. This means the bias is baked "
          f"in from initialization (e.g. actor head's bias term, or how "
          f"logits interact with this observation scale) -- NOT something "
          f"that develops during training. Check the actor layer's bias "
          f"initialization, or whether raw (non-normalized) observation "
          f"magnitudes are saturating the Tanh activations in a way that "
          f"pushes logits toward one action.")
elif max(action_counts.values()) / args.n_samples > 0.6:
    dominant = max(action_counts, key=action_counts.get)
    print(f"DIAGNOSIS: action={dominant} dominates even in a fresh "
          f"network, but it's not action=0 specifically. Still points to "
          f"an initialization-time bias rather than a training-dynamics "
          f"problem.")
else:
    print("DIAGNOSIS: fresh network's action distribution looks roughly "
          "uniform/reasonable -- NOT biased toward action=0 from "
          "initialization. If training still collapses to action=0, the "
          "bug is a LEARNING DYNAMICS problem (e.g. reward scale causing "
          "value/policy gradient issues, or a genuine local optimum "
          "where standing still avoids the -500 collision penalty), not "
          "a network-initialization problem. This supports revisiting "
          "reward shaping or GAE/entropy tuning, not env-side debugging.")