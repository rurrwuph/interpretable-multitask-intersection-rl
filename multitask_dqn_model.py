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

    def forward(self, obs):
        """
        obs: (batch, OBS_DIM) float tensor, layout
             [v_e, x_1,y_1,v_1,cos_1,sin_1, ..., x_5,...,sin_5]
 
        Returns: R(s,a,g) as (batch, n_actions, n_subtasks)
        """

        batch = obs.shape[0]

        ego = obs[:, :EGO_DIM]                       # (batch, 1)
        social = obs[:, EGO_DIM:]                     # (batch, 25)
        social = social.view(batch, N_SOCIAL, SOCIAL_DIM)  # (batch,5,5)

        ego_padded = torch.zeros(batch, SOCIAL_DIM, device=obs.device, dtype=obs.dtype)

        ego_padded[:, 0] = ego[:, 0]

        # Proposed Change
        # ego_padded[:, 0] = 0.0              # x_ego (origin)
        # ego_padded[:, 1] = 0.0              # y_ego (origin)
        # ego_padded[:, 2] = ego[:, 0]        # v_ego (velocity matches column 2)
        # ego_padded[:, 3] = 1.0              # cos(0)
        # ego_padded[:, 4] = 0.0              # sin(0)

        # stack all 6 slots and apply the SAME encoder to each
        # (batch, 6, SOCIAL_DIM)
        all_slots = torch.cat([ego_padded.unsqueeze(1), social], dim=1)
        encoded = self.slot_encoder(all_slots)         # (batch,6,slot_hidden)
        encoded_flat = encoded.view(batch, -1)          # (batch, 6*slot_hidden)
 
        trunk_out = self.trunk(encoded_flat)             # (batch, trunk_hidden)
 
        v = self.value_stream(trunk_out)                 # (batch, n_subtasks)
        a = self.advantage_stream(trunk_out).view(
            batch, self.n_actions, self.n_subtasks)       # (batch,n_act,n_sub)
 
        # dueling combine, per sub-task independently:
        # R(s,a,g)_k = V(s)_k + (A(s,a)_k - mean_a A(s,a)_k)
        a_mean = a.mean(dim=1, keepdim=True)              # (batch,1,n_sub)
        R = v.unsqueeze(1) + (a - a_mean)                 # (batch,n_act,n_sub)
        return R

    @staticmethod
    def masked_q(R, g):
        """Q(s,a;g) = g^T R(s,a,g), Eq. (4)-(5).
 
        R: (batch, n_actions, n_subtasks)
        g: (batch, n_subtasks)
        returns: (batch, n_actions)
        """
        # (batch, n_actions, n_subtasks) x (batch, n_subtasks, 1)
        # -> (batch, n_actions, 1) -> (batch, n_actions)
        return torch.bmm(R, g.unsqueeze(-1)).squeeze(-1)




# forward pass dimension check verified
