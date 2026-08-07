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

        
        # trial_mlp_head_0 = nn.Linear(64, 4)
        # trial_mlp_head_1 = nn.Linear(64, 4)
        # trial_mlp_head_2 = nn.Linear(64, 4)
        # trial_mlp_head_3 = nn.Linear(64, 4)
        # trial_mlp_head_4 = nn.Linear(64, 4)
        # trial_mlp_head_5 = nn.Linear(64, 4)
        # trial_mlp_head_6 = nn.Linear(64, 4)
        # trial_mlp_head_7 = nn.Linear(64, 4)
        # trial_mlp_head_8 = nn.Linear(64, 4)
        # trial_mlp_head_9 = nn.Linear(64, 4)
        # trial_mlp_head_10 = nn.Linear(64, 4)
        # trial_mlp_head_11 = nn.Linear(64, 4)
        # trial_mlp_head_12 = nn.Linear(64, 4)
        # trial_mlp_head_13 = nn.Linear(64, 4)
        # trial_mlp_head_14 = nn.Linear(64, 4)
        # trial_mlp_head_15 = nn.Linear(64, 4)
        # trial_mlp_head_16 = nn.Linear(64, 4)
        # trial_mlp_head_17 = nn.Linear(64, 4)
        # trial_mlp_head_18 = nn.Linear(64, 4)
        # trial_mlp_head_19 = nn.Linear(64, 4)
        # trial_mlp_head_20 = nn.Linear(64, 4)
        # trial_mlp_head_21 = nn.Linear(64, 4)
        # trial_mlp_head_22 = nn.Linear(64, 4)
        # trial_mlp_head_23 = nn.Linear(64, 4)
        # trial_mlp_head_24 = nn.Linear(64, 4)