import torch
import torch.nn as nn

N_SUBTASKS = 4
N_ACTIONS = 4
EGO_DIM = 1
SOCIAL_DIM = 5
N_SOCIAL = 5

OBS_DIM = EGO_DIM + N_SOCIAL*SOCIAL_DIM

class MultiTaskDQN(nn.Module):

    def __init__(self, obs_dim = OBS_DIM, n_actions = N_ACTIONS, n_subtasks= N_SUBTASKS, slot_hidden = 32, trunk_hidden=128,  stream_hidden=64):
        super().__init__()
        self.n_actions = n_actions
        self.n_subtasks = n_subtasks

        self.slot_encoder = nn.Sequential(
            nn.Linear(SOCIAL_DIM, slot_hidden),
            nn.ReLU(),
        )

        n_slots = 1+ N_SOCIAL

        self.trunk = nn.Sequential (
            nn.Linear(n_slots *slot_hidden, trunk_hidden),
            nn.ReLU(),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.ReLU(),
        )

         # stream 1: state-value-like term V(s) in R^{n_subtasks}
        # stream 2: advantage-like term A(s,a) in R^{n_actions x n_subtasks}

        self.value_stream = nn.Sequential(
            nn.Linear(trunk_hidden, stream_hidden),
            nn.ReLU(),
            nn.Linear(stream_hidden, n_subtasks),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(trunk_hidden, stream_hidden),
            nn.ReLU(),
            nn.Linear(stream_hidden, n_actions * n_subtasks),
        )

    