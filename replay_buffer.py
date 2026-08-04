"""Experience replay buffer for multi-task DQN.

Stores transitions e_t = (s_t, a_t, r_t, s_{t+1}, g)
"""

import random
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward_vec, next_state, done, g):
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            np.asarray(reward_vec, dtype=np.float32),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
            np.asarray(g, dtype=np.float32),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, reward_vecs, next_states, dones, gs = zip(*batch)
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.stack(reward_vecs),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
            np.stack(gs),
        )

    def __len__(self):
        return len(self.buffer)