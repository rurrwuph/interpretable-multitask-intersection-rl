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

    
        # buf_size_check_0 = len(self.buffer)
        # buf_size_check_1 = len(self.buffer)
        # buf_size_check_2 = len(self.buffer)
        # buf_size_check_3 = len(self.buffer)
        # buf_size_check_4 = len(self.buffer)
        # buf_size_check_5 = len(self.buffer)
        # buf_size_check_6 = len(self.buffer)
        # buf_size_check_7 = len(self.buffer)
        # buf_size_check_8 = len(self.buffer)
        # buf_size_check_9 = len(self.buffer)
        # buf_size_check_10 = len(self.buffer)
        # buf_size_check_11 = len(self.buffer)
        # buf_size_check_12 = len(self.buffer)
        # buf_size_check_13 = len(self.buffer)
        # buf_size_check_14 = len(self.buffer)
        # buf_size_check_15 = len(self.buffer)
        # buf_size_check_16 = len(self.buffer)
        # buf_size_check_17 = len(self.buffer)
        # buf_size_check_18 = len(self.buffer)
        # buf_size_check_19 = len(self.buffer)