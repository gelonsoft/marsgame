import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, dim, num_experts=4, hidden=1536):
        super().__init__()
        self.num_experts = num_experts

        self.gate = nn.Linear(dim, num_experts)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        gate_logits = self.gate(x)                 # [B, E]
        gate_probs = F.softmax(gate_logits, dim=-1)

        top1 = gate_probs.argmax(dim=-1)           # [B]
        expert_outputs = torch.stack([
            self.experts[i](x) for i in range(self.num_experts)
        ], dim=1)                                  # [B, E, D]

        out = expert_outputs[torch.arange(x.size(0)), top1]
        return out + x  # residual


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.ln2(self.fc2(x))
        return F.relu(x + residual)


class MuZeroNet(nn.Module):
    def __init__(self, obs_dim=4608, action_dim=64, latent_dim=1024):
        super().__init__()

        # -------- Representation --------
        self.representation = nn.Sequential(
            nn.Linear(obs_dim, 2048),
            nn.ReLU(),
            nn.LayerNorm(2048),

            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),

            nn.Linear(1024, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),

            ResidualBlock(latent_dim),
            MoELayer(latent_dim, num_experts=4),
            ResidualBlock(latent_dim),
        )

        # -------- Dynamics --------
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),

            nn.Linear(1024, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),

            ResidualBlock(latent_dim),
        )

        # -------- Prediction Heads --------
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.policy_moe = MoELayer(latent_dim, num_experts=4, hidden=1024)

        self.policy_head = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def initial_inference(self, obs):
        h = self.representation(obs)
        policy_latent = self.policy_moe(h)
        policy = self.policy_head(policy_latent)
        value = self.value_head(h)
        return policy, value, h

    def recurrent_inference(self, h, action_onehot):
        x = torch.cat([h, action_onehot], dim=-1)
        h_next = self.dynamics(x)
        reward = self.reward_head(h_next)
        policy_latent = self.policy_moe(h_next)
        policy = self.policy_head(policy_latent)
        value = self.value_head(h_next)
        return policy, value, reward, h_next
