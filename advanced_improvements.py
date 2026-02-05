"""
ADVANCED MUZERO TRAINER - IMPROVEMENTS 11-20
=============================================

Implements cutting-edge optimizations:
11. Multi-critic (phase-conditioned value heads)
12. Auxiliary prediction heads (self-supervised)
13. KL-penalty instead of PPO clipping
14. Mask-aware entropy
15. Action-value decomposition (Q = V + A)
16. Delayed-decision credit shaping
17. Per-head learning rates
18. Policy distillation from greedy heuristic
19. Generation-aware discount factor
20. Terminal-state value anchoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# 11. MULTI-CRITIC: Phase-Conditioned Value Heads
# ============================================================================
class MultiCriticValueNetwork(nn.Module):
    """
    Three separate value heads for different game phases.
    Each sees a stationary target distribution.
    
    Phases:
    - Early (gen 1-5): Engine building
    - Mid (gen 6-10): Transition
    - Late (gen 11-14): VP optimization
    """
    def __init__(self, latent_dim=4096):
        super().__init__()
        
        # Shared trunk
        self.shared_trunk = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
        )
        
        # Three separate critics
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
        
        # Blending network (learns optimal combination)
        self.blend_net = nn.Sequential(
            nn.Linear(512 + 3, 64),  # +3 for generation, terraform%, rounds_left
            nn.GELU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, h, generation=7, terraform_pct=0.5, rounds_left=7):
        """
        Args:
            h: [B, latent_dim] hidden state
            generation: current generation (1-14)
            terraform_pct: completion percentage (0-1)
            rounds_left: estimated rounds remaining
            
        Returns:
            value: [B, 1] blended value prediction
            phase_values: dict of individual phase predictions
        """
        B = h.shape[0]
        
        # Shared features
        shared = self.shared_trunk(h)
        
        # Individual phase predictions
        v_early = self.value_early(shared)
        v_mid = self.value_mid(shared)
        v_late = self.value_late(shared)
        
        # Phase indicators as features
        phase_features = torch.tensor([
            [generation / 14.0, terraform_pct, rounds_left / 14.0]
        ], dtype=torch.float32, device=h.device).repeat(B, 1)
        
        # Learn blend weights
        blend_input = torch.cat([shared, phase_features], dim=-1)
        weights = self.blend_net(blend_input)  # [B, 3]
        
        # Weighted combination
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
        """
        Select specific phase critic based on generation.
        Used during training to compute phase-specific losses.
        """
        shared = self.shared_trunk(h)
        
        if generation <= 5:
            return self.value_early(shared)
        elif generation <= 10:
            return self.value_mid(shared)
        else:
            return self.value_late(shared)


# ============================================================================
# 12. AUXILIARY PREDICTION HEADS (Self-Supervised)
# ============================================================================
class AuxiliaryPredictionHeads(nn.Module):
    """
    Predict game-state features to stabilize representation learning.
    These provide dense training signal beyond sparse rewards.
    """
    def __init__(self, latent_dim=4096):
        super().__init__()
        
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Predict next-generation metrics
        self.tr_predictor = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1)  # Predict TR delta
        )
        
        self.production_predictor = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 6)  # 6 production types (MC, Steel, Ti, Plant, Energy, Heat)
        )
        
        self.vp_delta_predictor = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1)  # Predict VP change
        )
        
        # Binary predictions
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
        """
        Predict auxiliary targets from hidden state.
        
        Returns:
            dict of predictions
        """
        features = self.trunk(h)
        
        return {
            'tr_delta': self.tr_predictor(features),
            'production': self.production_predictor(features),
            'vp_delta': self.vp_delta_predictor(features),
            'can_play_card': self.can_play_card(features),
            'resource_sufficient': self.resource_sufficient(features)
        }
    
    def compute_auxiliary_loss(self, predictions, targets):
        """
        Compute auxiliary losses.
        
        Args:
            predictions: dict from forward()
            targets: dict of ground truth values
            
        Returns:
            total_aux_loss: weighted sum of auxiliary losses
        """
        loss = torch.tensor(0.0, device=predictions['tr_delta'].device)
        
        # Regression losses
        if 'tr_delta' in targets:
            loss += F.mse_loss(predictions['tr_delta'], targets['tr_delta'])
        
        if 'production' in targets:
            loss += F.mse_loss(predictions['production'], targets['production'])
        
        if 'vp_delta' in targets:
            loss += F.mse_loss(predictions['vp_delta'], targets['vp_delta'])
        
        # Binary classification losses
        if 'can_play_card' in targets:
            loss += F.binary_cross_entropy(
                predictions['can_play_card'], 
                targets['can_play_card']
            )
        
        if 'resource_sufficient' in targets:
            loss += F.binary_cross_entropy(
                predictions['resource_sufficient'],
                targets['resource_sufficient']
            )
        
        return loss


# ============================================================================
# 13. KL-Penalty instead of PPO Clipping
# ============================================================================
class AdaptiveKLController:
    """
    Adaptive KL penalty to smoothly constrain policy updates.
    Better than hard PPO clipping for large masked action spaces.
    
    Algorithm:
    - If KL > target: increase β (stronger penalty)
    - If KL < target: decrease β (weaker penalty)
    """
    def __init__(self, target_kl=0.01, beta_init=0.1, beta_min=0.01, beta_max=1.0):
        self.target_kl = target_kl
        self.beta = beta_init
        self.beta_min = beta_min
        self.beta_max = beta_max
        
        # Adaptation parameters
        self.adapt_factor = 1.5
    
    def update(self, kl_div):
        """
        Update β based on observed KL divergence.
        
        Args:
            kl_div: measured KL(π_old || π_new)
        """
        if kl_div > self.target_kl * 1.5:
            # KL too high → increase penalty
            self.beta = min(self.beta * self.adapt_factor, self.beta_max)
        elif kl_div < self.target_kl / 1.5:
            # KL too low → decrease penalty
            self.beta = max(self.beta / self.adapt_factor, self.beta_min)
        
        return self.beta
    
    def compute_kl_penalty(self, old_logprobs, new_logprobs):
        """
        Compute KL penalty term.
        
        Args:
            old_logprobs: [B] log π_old(a|s)
            new_logprobs: [B] log π_new(a|s)
            
        Returns:
            kl_penalty: scalar penalty term
            kl_div: scalar KL divergence value
        """
        # KL(old || new) = E[log(old) - log(new)]
        kl_div = (old_logprobs - new_logprobs).mean()
        kl_penalty = self.beta * kl_div
        
        return kl_penalty, kl_div.item()


# ============================================================================
# 14. MASK-AWARE ENTROPY
# ============================================================================
def compute_masked_entropy(logits, mask):
    """
    Compute entropy only over LEGAL actions.
    
    Problem: Standard entropy includes masked-out actions,
    diluting gradient signal.
    
    Args:
        logits: [B, A] policy logits
        mask: [B, A] binary mask (1 = legal, 0 = illegal)
        
    Returns:
        entropy: [B] entropy over legal actions only
    """
    # Mask out illegal actions before softmax
    masked_logits = logits.clone()
    masked_logits[mask == 0] = -1e9
    
    # Softmax over legal actions
    probs = F.softmax(masked_logits, dim=-1)
    
    # Compute entropy only where mask = 1
    log_probs = F.log_softmax(masked_logits, dim=-1)
    
    # Entropy = -Σ p(a) log p(a) for legal actions
    entropy = -(probs * log_probs * mask).sum(dim=-1)
    
    # Normalize by number of legal actions (optional)
    n_legal = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    # entropy = entropy / n_legal  # Uncomment for normalized entropy
    
    return entropy


def compute_masked_entropy_factored(policy_heads, head_masks):
    """
    Compute mask-aware entropy for factored action heads.
    
    Args:
        policy_heads: list of [B, head_dim] logits
        head_masks: list of [B, head_dim] binary masks
        
    Returns:
        total_entropy: scalar
        per_head_entropy: list of scalars
    """
    total_entropy = torch.tensor(0.0, device=policy_heads[0].device)
    per_head_entropy = []
    
    for logits, mask in zip(policy_heads, head_masks):
        head_ent = compute_masked_entropy(logits, mask).mean()
        total_entropy += head_ent
        per_head_entropy.append(head_ent.item())
    
    return total_entropy, per_head_entropy


# ============================================================================
# 15. ACTION-VALUE DECOMPOSITION (Dueling Architecture)
# ============================================================================
class DuelingValueHead(nn.Module):
    """
    Q(s,a) = V(s) + A(s,a)
    
    Separates state value from action advantages.
    Reduces variance in advantage estimation.
    """
    def __init__(self, latent_dim=4096, num_actions=64):
        super().__init__()
        
        # State value stream
        self.value_stream = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_actions)
        )
    
    def forward(self, h):
        """
        Args:
            h: [B, latent_dim] hidden state
            
        Returns:
            q_values: [B, num_actions]
            value: [B, 1]
            advantages: [B, num_actions]
        """
        value = self.value_stream(h)
        advantages = self.advantage_stream(h)
        
        # Q = V + (A - mean(A))
        # Subtract mean for identifiability
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
        
        return q_values, value, advantages


# ============================================================================
# 16. DELAYED-DECISION CREDIT SHAPING
# ============================================================================
class CreditShapingRewards:
    """
    Add synthetic intermediate rewards for long-horizon credit assignment.
    Zero out once terraforming is complete.
    """
    def __init__(self, shaping_weight=0.1):
        self.shaping_weight = shaping_weight
    
    def compute_shaping_reward(self, state_before, state_after, terraform_complete=False):
        """
        Compute intrinsic reward for intermediate progress.
        
        Args:
            state_before: dict of game state before action
            state_after: dict of game state after action
            terraform_complete: bool, True if all global params maxed
            
        Returns:
            shaped_reward: float
        """
        if terraform_complete:
            # No shaping in endgame (pure VP optimization)
            return 0.0
        
        reward = 0.0
        
        # Production diversity bonus
        prod_types_before = self._count_production_types(state_before)
        prod_types_after = self._count_production_types(state_after)
        reward += 0.5 * (prod_types_after - prod_types_before)
        
        # Card diversity bonus (more card types = more options)
        card_types_before = len(set(state_before.get('tableau', [])))
        card_types_after = len(set(state_after.get('tableau', [])))
        reward += 0.3 * (card_types_after - card_types_before)
        
        # Payment flexibility (steel/titanium production)
        payment_flex_before = (
            state_before.get('steelProduction', 0) + 
            state_before.get('titaniumProduction', 0)
        )
        payment_flex_after = (
            state_after.get('steelProduction', 0) + 
            state_after.get('titaniumProduction', 0)
        )
        reward += 0.2 * (payment_flex_after - payment_flex_before)
        
        return reward * self.shaping_weight
    
    def _count_production_types(self, state):
        """Count how many production types are > 0"""
        prod_keys = [
            'megaCreditProduction', 'steelProduction', 'titaniumProduction',
            'plantProduction', 'energyProduction', 'heatProduction'
        ]
        return sum(1 for k in prod_keys if state.get(k, 0) > 0)


# ============================================================================
# 17. PER-HEAD LEARNING RATES
# ============================================================================
class PerHeadOptimizer:
    """
    Different learning rates for different policy heads.
    
    Head difficulty:
    - Head 0 (action_type): HARD, strategic → lower LR
    - Head 1 (object_id): MEDIUM → medium LR  
    - Head 2 (payment_id): EASY, tactical → higher LR
    - Head 3 (extra_param): EASY → highest LR
    """
    def __init__(self, model, base_lr=3e-4):
        self.model = model
        self.base_lr = base_lr
        
        # LR multipliers per head
        self.lr_multipliers = {
            'head_0': 1.0,   # 3e-4 (strategic, hard)
            'head_1': 1.5,   # 4.5e-4 (medium)
            'head_2': 2.0,   # 6e-4 (tactical, easier)
            'head_3': 2.5,   # 7.5e-4 (easiest)
        }
        
        # Group parameters by head
        param_groups = self._create_param_groups()
        
        # Create optimizer with per-group LRs
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
    
    def _create_param_groups(self):
        """Create parameter groups with different LRs"""
        param_groups = []
        
        # Policy heads (different LRs)
        for i, head in enumerate(self.model.policy_heads):
            head_name = f'head_{i}'
            lr = self.base_lr * self.lr_multipliers[head_name]
            param_groups.append({
                'params': head.parameters(),
                'lr': lr,
                'name': head_name
            })
        
        # Value head and other components (base LR)
        other_params = []
        policy_head_ids = {id(p) for head in self.model.policy_heads for p in head.parameters()}
        
        for name, param in self.model.named_parameters():
            if id(param) not in policy_head_ids and param.requires_grad:
                other_params.append(param)
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.base_lr,
                'name': 'other'
            })
        
        return param_groups
    
    def step(self):
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()


# ============================================================================
# 18. POLICY DISTILLATION FROM GREEDY HEURISTIC
# ============================================================================
class GreedyHeuristicPolicy:
    """
    Simple scripted policy for pretraining.
    
    Rules:
    - Early game (gen 1-5): Prefer production cards
    - Mid game (gen 6-10): Balanced
    - Late game (gen 11+): Prefer TR and VP cards
    """
    def __init__(self):
        self.production_card_names = [
            'Solar Power', 'Geothermal Power', 'Mine', 'Strip Mine',
            'Ironworks', 'Steelworks', 'Titanium Mine'
        ]
        
        self.tr_card_names = [
            'Asteroid', 'Comet', 'Ice Asteroid', 'Water Import'
        ]
        
        self.vp_card_names = [
            'Ganymede Colony', 'Space Port', 'Capitol', 'University'
        ]
    
    def select_action(self, legal_actions, generation, game_state):
        """
        Select action using simple heuristic.
        
        Returns:
            action_index: int
        """
        # Categorize legal actions
        production_actions = []
        tr_actions = []
        vp_actions = []
        other_actions = []
        
        for i, action in enumerate(legal_actions):
            card_name = action.get('card', {}).get('name', '')
            
            if any(prod in card_name for prod in self.production_card_names):
                production_actions.append(i)
            elif any(tr in card_name for tr in self.tr_card_names):
                tr_actions.append(i)
            elif any(vp in card_name for vp in self.vp_card_names):
                vp_actions.append(i)
            else:
                other_actions.append(i)
        
        # Phase-based selection
        if generation <= 5:  # Early: production
            if production_actions:
                return np.random.choice(production_actions)
        elif generation >= 11:  # Late: TR and VP
            if tr_actions:
                return np.random.choice(tr_actions)
            if vp_actions:
                return np.random.choice(vp_actions)
        
        # Fallback: random legal action
        return np.random.choice(len(legal_actions))
    
    def collect_demonstrations(self, env, num_episodes=100):
        """
        Collect (state, action) pairs from heuristic policy.
        
        Returns:
            demonstrations: list of (obs, action_index) tuples
        """
        demonstrations = []
        
        for _ in range(num_episodes):
            obs = env.reset()
            done = False
            
            while not done:
                legal_actions = env.get_legal_actions()
                generation = env.get_generation()
                state = env.get_state()
                
                action_idx = self.select_action(legal_actions, generation, state)
                demonstrations.append((obs, action_idx))
                
                obs, reward, done, info = env.step(action_idx)
        
        return demonstrations


def pretrain_policy(model, demonstrations, epochs=3, batch_size=256):
    """
    Pretrain policy via behavior cloning on heuristic demonstrations.
    
    Args:
        model: MuZeroNet
        demonstrations: list of (obs, action_idx) tuples
        epochs: number of training epochs
        
    Returns:
        model with pretrained policy
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        np.random.shuffle(demonstrations)
        total_loss = 0.0
        
        for i in range(0, len(demonstrations), batch_size):
            batch = demonstrations[i:i+batch_size]
            obs_batch = torch.stack([torch.tensor(obs) for obs, _ in batch]).to(DEVICE)
            action_batch = torch.tensor([act for _, act in batch]).to(DEVICE)
            
            # Forward pass
            policy_heads, _, _ = model.initial_inference(obs_batch)
            
            # Behavior cloning loss (cross-entropy)
            # Note: This assumes flat action space; adapt for factored
            loss = F.cross_entropy(policy_heads[0], action_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Pretrain epoch {epoch+1}/{epochs}, loss: {total_loss/len(demonstrations):.4f}")
    
    return model


# ============================================================================
# 19. GENERATION-AWARE DISCOUNT FACTOR
# ============================================================================
class GenerationAwareDiscount:
    """
    Variable discount factor based on game phase.
    
    Early game: γ = 0.995 (more far-sighted)
    Mid game: γ = 0.99
    Late game: γ = 0.95 (more myopic, reduce variance)
    """
    def __init__(self):
        self.schedules = [
            (5, 0.995),   # Gen 1-5
            (10, 0.99),   # Gen 6-10
            (14, 0.95),   # Gen 11-14
        ]
    
    def get_gamma(self, generation):
        """Get discount factor for current generation"""
        for max_gen, gamma in self.schedules:
            if generation <= max_gen:
                return gamma
        return 0.95  # Default
    
    def compute_discounted_return(self, rewards, generations):
        """
        Compute return with variable discount.
        
        Args:
            rewards: [T] reward sequence
            generations: [T] generation for each timestep
            
        Returns:
            returns: [T] discounted returns
        """
        T = len(rewards)
        returns = np.zeros(T, dtype=np.float32)
        
        running_return = 0.0
        for t in reversed(range(T)):
            gamma = self.get_gamma(generations[t])
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        return returns


# ============================================================================
# 20. TERMINAL-STATE VALUE ANCHORING
# ============================================================================
class TerminalValueAnchor:
    """
    Enforce V(s_terminal) = R_terminal with extra supervised loss.
    Prevents value drift and anchors critic backward through time.
    """
    def __init__(self, anchor_weight=5.0):
        self.anchor_weight = anchor_weight
    
    def compute_anchor_loss(self, value_pred, target_value, is_terminal):
        """
        Compute anchoring loss for terminal states.
        
        Args:
            value_pred: [B] predicted values
            target_value: [B] target values (exact final VP for terminals)
            is_terminal: [B] bool mask for terminal states
            
        Returns:
            anchor_loss: scalar
        """
        # Only compute on terminal states
        terminal_mask = is_terminal.float()
        n_terminal = terminal_mask.sum()
        
        if n_terminal == 0:
            return torch.tensor(0.0, device=value_pred.device)
        
        # MSE on terminal states only
        terminal_error = (value_pred - target_value) ** 2
        anchor_loss = (terminal_error * terminal_mask).sum() / n_terminal
        
        return self.anchor_weight * anchor_loss


# ============================================================================
# USAGE EXAMPLE: INTEGRATED TRAINING LOOP
# ============================================================================
def advanced_train_step(
    model,
    batch,
    multi_critic,
    aux_heads,
    kl_controller,
    credit_shaper,
    gamma_scheduler,
    terminal_anchor,
    optimizer,
    step
):
    """
    Training step with all advanced improvements.
    
    Returns:
        losses: dict of loss components
        metrics: dict of training metrics
    """
    obs, actions, rewards, next_obs, dones, old_logprobs = batch
    generation = 7  # Would come from batch metadata
    terraform_pct = 0.5
    
    # Forward pass
    policy_heads, _, h = model.initial_inference(obs)
    
    # ================================================================
    # 11. Multi-critic value prediction
    # ================================================================
    value, phase_values = multi_critic(h, generation, terraform_pct)
    
    # ================================================================
    # 12. Auxiliary predictions
    # ================================================================
    aux_preds = aux_heads(h)
    aux_targets = {
        'tr_delta': torch.zeros_like(rewards),  # Would be real targets
        'vp_delta': rewards,
    }
    aux_loss = aux_heads.compute_auxiliary_loss(aux_preds, aux_targets)
    
    # ================================================================
    # 13. KL penalty (instead of PPO clipping)
    # ================================================================
    # Compute new log probs
    new_logprobs = torch.log_softmax(policy_heads[0], dim=-1)
    new_logprobs = new_logprobs.gather(1, actions.unsqueeze(1)).squeeze()
    
    kl_penalty, kl_div = kl_controller.compute_kl_penalty(old_logprobs, new_logprobs)
    
    # ================================================================
    # 14. Mask-aware entropy
    # ================================================================
    mask = torch.ones_like(policy_heads[0])  # Would be real mask
    entropy, _ = compute_masked_entropy_factored(policy_heads, [mask] * len(policy_heads))
    
    # ================================================================
    # 19. Generation-aware discount
    # ================================================================
    gamma = gamma_scheduler.get_gamma(generation)
    
    # ================================================================
    # 20. Terminal value anchoring
    # ================================================================
    anchor_loss = terminal_anchor.compute_anchor_loss(value.squeeze(), rewards, dones)
    
    # ================================================================
    # Total loss
    # ================================================================
    policy_loss = torch.tensor(0.0, device=DEVICE)  # Would be real policy loss
    value_loss = F.mse_loss(value.squeeze(), rewards)
    
    total_loss = (
        policy_loss + 
        value_loss + 
        0.1 * aux_loss +  # Auxiliary weight
        kl_penalty +
        anchor_loss -
        0.01 * entropy  # Entropy bonus
    )
    
    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # Update KL controller
    kl_controller.update(kl_div)
    
    return {
        'total_loss': total_loss.item(),
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'aux_loss': aux_loss.item(),
        'kl_penalty': kl_penalty.item(),
        'kl_div': kl_div,
        'anchor_loss': anchor_loss.item(),
        'entropy': entropy.item(),
    }, {
        'gamma': gamma,
        'kl_beta': kl_controller.beta,
    }


if __name__ == "__main__":
    print("Advanced MuZero trainer with improvements 11-20 loaded successfully!")