import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):
    """IMPROVED: Better MoE with top-k routing"""
    def __init__(self, dim, num_experts=4, hidden=1536, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Linear(dim, num_experts)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),  # Changed from ReLU for smoother gradients
                nn.Dropout(0.1),  # NEW: Dropout for regularization
                nn.Linear(hidden, dim)
            ) for _ in range(num_experts)
        ])
        
        # NEW: Load balancing loss weight
        self.load_balance_weight = 0.01

    def forward(self, x):
        gate_logits = self.gate(x)  # [B, E]
        gate_probs = F.softmax(gate_logits, dim=-1)

        # NEW: Top-k expert routing
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Compute expert outputs
        expert_outputs = torch.stack([
            self.experts[i](x) for i in range(self.num_experts)
        ], dim=1)  # [B, E, D]

        # Combine top-k experts
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]
            expert_weight = top_k_probs[:, k].unsqueeze(-1)
            expert_out = expert_outputs[torch.arange(x.size(0)), expert_idx]
            out += expert_weight * expert_out

        return out + x  # residual


class ResidualBlock(nn.Module):
    """IMPROVED: Better residual block with pre-normalization"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)  # Wider hidden layer
        self.ln2 = nn.LayerNorm(dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-normalization for better gradient flow
        residual = x
        x = self.ln1(x)
        x = self.fc1(x)
        x = F.gelu(x)  # GELU instead of ReLU
        x = self.dropout(x)
        x = self.ln2(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class AttentionBlock(nn.Module):
    """NEW: Self-attention for capturing long-range dependencies"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [B, D]
        residual = x
        x = self.ln(x)
        x_seq = x.unsqueeze(1)  # [B, 1, D]
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        x = self.dropout(attn_out.squeeze(1))
        return x + residual


class MuZeroNet(nn.Module):
    def __init__(self, obs_dim=4608, action_dim=64, latent_dim=1024):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # -------- Representation Network --------
        # IMPROVED: Better architecture with gradual dimension reduction
        self.representation = nn.Sequential(
            nn.Linear(obs_dim, 2048),
            nn.GELU(),
            nn.LayerNorm(2048),
            nn.Dropout(0.1),

            ResidualBlock(2048, dropout=0.1),
            
            nn.Linear(2048, 1536),
            nn.GELU(),
            nn.LayerNorm(1536),
            nn.Dropout(0.1),

            ResidualBlock(1536, dropout=0.1),
            
            nn.Linear(1536, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),

            ResidualBlock(latent_dim, dropout=0.1),
            MoELayer(latent_dim, num_experts=4, top_k=2),
            ResidualBlock(latent_dim, dropout=0.1),
        )

        # -------- Dynamics Network --------
        # IMPROVED: Better dynamics with attention
        self.dynamics_input = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.1)
        )
        
        self.dynamics_core = nn.Sequential(
            ResidualBlock(1024, dropout=0.1),
            AttentionBlock(1024, num_heads=8),  # NEW: Attention
            ResidualBlock(1024, dropout=0.1),
        )
        
        self.dynamics_output = nn.Sequential(
            nn.Linear(1024, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim)
        )

        # -------- Prediction Heads --------
        # IMPROVED: Separate prediction networks for better learning
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 1)
        )

        # Policy head with MoE
        self.policy_moe = MoELayer(latent_dim, num_experts=4, hidden=1024, top_k=2)
        
        self.policy_head = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, action_dim)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 1),
            nn.Tanh()  # NEW: Tanh to bound value predictions
        )
        
        # NEW: Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Proper weight initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def initial_inference(self, obs):
        """
        Initial inference from observation
        Returns: policy logits, value, hidden state
        """
        h = self.representation(obs)
        
        # Get policy
        policy_latent = self.policy_moe(h)
        policy = self.policy_head(policy_latent)
        
        # Get value
        value = self.value_head(h)
        
        return policy, value, h

    def recurrent_inference(self, h, action_onehot):
        """
        Recurrent inference from hidden state and action
        Returns: policy logits, value, reward, next hidden state
        """
        # Dynamics step
        x = torch.cat([h, action_onehot], dim=-1)
        x = self.dynamics_input(x)
        x = self.dynamics_core(x)
        h_next = self.dynamics_output(x)
        
        # Add residual connection for stability
        h_next = h_next + h
        
        # Predict reward
        reward = self.reward_head(h_next)
        
        # Get policy
        policy_latent = self.policy_moe(h_next)
        policy = self.policy_head(policy_latent)
        
        # Get value
        value = self.value_head(h_next)
        
        return policy, value, reward, h_next
    
    def get_num_parameters(self):
        """Utility to count parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)