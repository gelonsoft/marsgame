import torch
import torch.nn as nn
import torch.nn.functional as F

class MuZeroNet(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.representation = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )

        self.dynamics = nn.Sequential(
            nn.Linear(256 + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        self.reward_head = nn.Linear(256, 1)
        self.policy_head = nn.Linear(256, action_dim)
        self.value_head = nn.Linear(256, 1)

    def initial_inference(self, obs):
        h = self.representation(obs)
        return self.policy_head(h), self.value_head(h), h

    def recurrent_inference(self, h, action_onehot):
        x = torch.cat([h, action_onehot], dim=-1)
        h_next = self.dynamics(x)
        reward = self.reward_head(h_next)
        policy = self.policy_head(h_next)
        value = self.value_head(h_next)
        return policy, value, reward, h_next
