"""
FINAL MUZERO NETWORK - ALL 20 IMPROVEMENTS INTEGRATED
======================================================

Complete implementation with:
- Multi-critic (phase-conditioned value heads)
- Auxiliary prediction heads
- Dueling architecture (Q = V + A)
- Improved architecture with all enhancements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from factored_actions import FACTORED_ACTION_DIMS


class MoELayer(nn.Module):
    """Mixture of Experts with top-k routing"""
    def __init__(self, dim, num_experts=4, hidden=1536, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, dim)
            ) for _ in range(num_experts)
        ])
        self.load_balance_weight = 0.01

    def forward(self, x):
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        expert_outputs = torch.stack([
            self.experts[i](x) for i in range(self.num_experts)
        ], dim=1)
        
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]
            expert_weight = top_k_probs[:, k].unsqueeze(-1)
            expert_out = expert_outputs[torch.arange(x.size(0)), expert_idx]
            out += expert_weight * expert_out
        
        return out + x


class ResidualBlock(nn.Module):
    """Residual block with pre-normalization"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.ln2 = nn.LayerNorm(dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.ln1(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.ln2(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class AttentionBlock(nn.Module):
    """Self-attention block"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.ln(x)
        x_seq = x.unsqueeze(1)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        x = self.dropout(attn_out.squeeze(1))
        return x + residual


# ============================================================================
# IMPROVEMENT 11: Multi-Critic (Phase-Conditioned Value Heads)
# ============================================================================
class MultiCriticValueNetwork(nn.Module):
    """
    Three separate value heads for different game phases.
    Each sees a stationary target distribution.
    """
    def __init__(self, latent_dim=4096):
        super().__init__()
        
        self.shared_trunk = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
        )
        
        # Three phase-specific critics
        self.value_early = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
        self.value_mid = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
        self.value_late = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
        # Blending network
        self.blend_net = nn.Sequential(
            nn.Linear(512 + 3, 64),
            nn.GELU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, h, generation=7, terraform_pct=0.5, rounds_left=7):
        B = h.shape[0]
        shared = self.shared_trunk(h)
        
        v_early = self.value_early(shared)
        v_mid = self.value_mid(shared)
        v_late = self.value_late(shared)
        
        phase_features = torch.tensor([
            [generation / 14.0, terraform_pct, rounds_left / 14.0]
        ], dtype=torch.float32, device=h.device).repeat(B, 1)
        
        blend_input = torch.cat([shared, phase_features], dim=-1)
        weights = self.blend_net(blend_input)
        
        value = (
            weights[:, 0:1] * v_early +
            weights[:, 1:2] * v_mid +
            weights[:, 2:3] * v_late
        )
        
        return value, {
            'early': v_early,
            'mid': v_mid,
            'late': v_late,
            'weights': weights
        }
    
    def get_phase_value(self, h, generation):
        shared = self.shared_trunk(h)
        if generation <= 5:
            return self.value_early(shared)
        elif generation <= 10:
            return self.value_mid(shared)
        else:
            return self.value_late(shared)


# ============================================================================
# IMPROVEMENT 12: Auxiliary Prediction Heads
# ============================================================================
class AuxiliaryPredictionHeads(nn.Module):
    """Predict game-state features for self-supervised learning"""
    def __init__(self, latent_dim=4096):
        super().__init__()
        
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.tr_predictor = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        self.production_predictor = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 6)
        )
        
        self.vp_delta_predictor = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        self.can_play_card = nn.Sequential(
            nn.Linear(512, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.resource_sufficient = nn.Sequential(
            nn.Linear(512, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, h):
        features = self.trunk(h)
        return {
            'tr_delta': self.tr_predictor(features),
            'production': self.production_predictor(features),
            'vp_delta': self.vp_delta_predictor(features),
            'can_play_card': self.can_play_card(features),
            'resource_sufficient': self.resource_sufficient(features)
        }
    
    def compute_auxiliary_loss(self, predictions, targets):
        loss = torch.tensor(0.0, device=predictions['tr_delta'].device)
        
        if 'tr_delta' in targets:
            loss += F.mse_loss(predictions['tr_delta'], targets['tr_delta'])
        if 'production' in targets:
            loss += F.mse_loss(predictions['production'], targets['production'])
        if 'vp_delta' in targets:
            loss += F.mse_loss(predictions['vp_delta'], targets['vp_delta'])
        if 'can_play_card' in targets:
            loss += F.binary_cross_entropy(predictions['can_play_card'], targets['can_play_card'])
        if 'resource_sufficient' in targets:
            loss += F.binary_cross_entropy(predictions['resource_sufficient'], targets['resource_sufficient'])
        
        return loss


# ============================================================================
# IMPROVEMENT 15: Dueling Architecture (Q = V + A)
# ============================================================================
class DuelingValueHead(nn.Module):
    """Q(s,a) = V(s) + A(s,a) for lower variance"""
    def __init__(self, latent_dim=4096, num_actions=64):
        super().__init__()
        
        self.value_stream = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_actions)
        )
    
    def forward(self, h):
        value = self.value_stream(h)
        advantages = self.advantage_stream(h)
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values, value, advantages


# ============================================================================
# MAIN MUZERO NETWORK - FINAL VERSION
# ============================================================================
class MuZeroNetFinal(nn.Module):
    """
    Complete MuZero network with all 20 improvements integrated.
    """
    def __init__(self, obs_dim=512, action_dim=64, latent_dim=4096, use_dueling=False):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.use_dueling = use_dueling
        
        # Factored action dimensions
        from factored_actions import FACTORED_ACTION_DIMS
        self.factored_action_dims = FACTORED_ACTION_DIMS
        self.factored_action_total = sum(FACTORED_ACTION_DIMS)

        # -------- Representation Network --------
        self.representation = nn.Sequential(
            nn.Linear(obs_dim, 8192),
            nn.GELU(),
            nn.LayerNorm(8192),
            nn.Dropout(0.1),
            ResidualBlock(8192, dropout=0.1),
            
            nn.Linear(8192, 1536),
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
        self.dynamics_input = nn.Sequential(
            nn.Linear(latent_dim + self.factored_action_total, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.1)
        )
        
        self.dynamics_core = nn.Sequential(
            ResidualBlock(1024, dropout=0.1),
            AttentionBlock(1024, num_heads=8),
            ResidualBlock(1024, dropout=0.1),
        )
        
        self.dynamics_output = nn.Sequential(
            nn.Linear(1024, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim)
        )

        # -------- Reward Head --------
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

        # -------- Policy Heads --------
        self.policy_moe = MoELayer(latent_dim, num_experts=4, hidden=1024, top_k=2)
        
        self.policy_heads = nn.ModuleList()
        for head_dim in self.factored_action_dims:
            self.policy_heads.append(nn.Sequential(
                nn.Linear(latent_dim, 512),
                nn.GELU(),
                nn.LayerNorm(512),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.LayerNorm(256),
                nn.Linear(256, head_dim)
            ))

        # -------- Value Heads --------
        # IMPROVEMENT 11: Multi-critic
        self.value_head = MultiCriticValueNetwork(latent_dim)
        
        # IMPROVEMENT 15: Optional dueling head
        if use_dueling:
            self.dueling_head = DuelingValueHead(latent_dim, action_dim)
        
        # IMPROVEMENT 12: Auxiliary heads
        self.aux_heads = AuxiliaryPredictionHeads(latent_dim)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def initial_inference(self, obs, generation=7, terraform_pct=0.5, rounds_left=7):
        """
        Initial inference from observation.
        
        Returns:
            policy_heads: list of 4 policy head logits
            value: blended value prediction
            phase_values: dict of phase-specific values
            aux_preds: dict of auxiliary predictions
            h: hidden state
            dueling_outputs: (optional) Q-values, V, A
        """
        h = self.representation(obs)
        
        # Policy heads
        policy_latent = self.policy_moe(h)
        policy_heads = [head(policy_latent) for head in self.policy_heads]
        
        # Multi-critic value
        value, phase_values = self.value_head(h, generation, terraform_pct, rounds_left)
        
        # Auxiliary predictions
        aux_preds = self.aux_heads(h)
        
        # Optional dueling
        dueling_outputs = None
        if self.use_dueling:
            q_values, state_value, advantages = self.dueling_head(h)
            dueling_outputs = (q_values, state_value, advantages)
        
        return policy_heads, value, phase_values, aux_preds, h, dueling_outputs

    def recurrent_inference(self, h, action_factored, generation=7, terraform_pct=0.5, rounds_left=7):
        """
        Recurrent inference from hidden state and factored action.
        
        Parameters:
            h: [B, latent_dim]
            action_factored: [B, sum(FACTORED_ACTION_DIMS)]
            
        Returns:
            policy_heads, value, phase_values, aux_preds, reward, h_next, dueling_outputs
        """
        # Dynamics step
        x = torch.cat([h, action_factored], dim=-1)
        x = self.dynamics_input(x)
        x = self.dynamics_core(x)
        h_next = self.dynamics_output(x)
        h_next = h_next + h  # Residual connection
        
        # Predict reward
        reward = self.reward_head(h_next)
        
        # Policy heads
        policy_latent = self.policy_moe(h_next)
        policy_heads = [head(policy_latent) for head in self.policy_heads]
        
        # Value
        value, phase_values = self.value_head(h_next, generation, terraform_pct, rounds_left)
        
        # Auxiliary
        aux_preds = self.aux_heads(h_next)
        
        # Optional dueling
        dueling_outputs = None
        if self.use_dueling:
            q_values, state_value, advantages = self.dueling_head(h_next)
            dueling_outputs = (q_values, state_value, advantages)
        
        return policy_heads, value, phase_values, aux_preds, reward, h_next, dueling_outputs
    
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_masked_entropy(logits, mask):
    """
    IMPROVEMENT 14: Mask-aware entropy computation.
    Compute entropy only over LEGAL actions.
    """
    masked_logits = logits.clone()
    masked_logits[mask == 0] = -1e9
    
    probs = F.softmax(masked_logits, dim=-1)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    
    entropy = -(probs * log_probs * mask).sum(dim=-1)
    return entropy


def compute_masked_entropy_factored(policy_heads, head_masks):
    """Compute mask-aware entropy for all factored heads"""
    total_entropy = torch.tensor(0.0, device=policy_heads[0].device)
    per_head_entropy = []
    
    for logits, mask in zip(policy_heads, head_masks):
        head_ent = compute_masked_entropy(logits, mask).mean()
        total_entropy += head_ent
        per_head_entropy.append(head_ent.item())
    
    return total_entropy, per_head_entropy


if __name__ == "__main__":
    # Test the network
    print("Testing MuZeroNetFinal...")
    
    model = MuZeroNetFinal(obs_dim=512, action_dim=64, latent_dim=4096, use_dueling=True)
    print(f"Total parameters: {model.get_num_parameters():,}")
    
    # Test initial inference
    obs = torch.randn(32, 512)
    policy_heads, value, phase_values, aux_preds, h, dueling_outputs = model.initial_inference(
        obs, generation=7, terraform_pct=0.5
    )
    
    print(f"Policy heads: {[p.shape for p in policy_heads]}")
    print(f"Value: {value.shape}")
    print(f"Phase values: {list(phase_values.keys())}")
    print(f"Aux predictions: {list(aux_preds.keys())}")
    print(f"Hidden state: {h.shape}")
    if dueling_outputs:
        print(f"Dueling Q-values: {dueling_outputs[0].shape}")
    
    print("\nMuZeroNetFinal initialized successfully!")