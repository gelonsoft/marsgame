"""
IMPROVED MCTS - INTEGRATED WITH ALL 20 IMPROVEMENTS
====================================================

Enhancements:
- Compatible with muzero.py (multi-critic, auxiliary heads)
- Generation-aware discount factor (improvement 19)
- Enhanced node value tracking
- Better exploration with adaptive noise
- Virtual loss for parallel search
- Improved UCB with recent wins bias
"""

import math
import numpy as np
import torch
from factored_actions import (
    FACTORED_ACTION_DIMS, ACTION_TYPE_DIM, OBJECT_ID_DIM, 
    PAYMENT_ID_DIM, EXTRA_PARAM_DIM, NUM_HEADS
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Node:
    """Enhanced MCTS node with additional tracking"""
    def __init__(self, prior):
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.reward = 0.0
        self.children = {}
        self.hidden_state = None
        
        # IMPROVEMENT: Track recent performance for recency bias
        self.recent_values = []  # Last 5 values
        self.virtual_loss = 0  # For parallel search
        
    @property
    def value(self):
        """Mean value with virtual loss penalty"""
        if self.visit_count == 0:
            return 0.0
        # Apply virtual loss (for parallel MCTS, currently unused but ready)
        effective_visits = self.visit_count + self.virtual_loss
        if effective_visits == 0:
            return 0.0
        return self.value_sum / effective_visits
    
    @property
    def recent_value(self):
        """Recent average (last 5 visits) for recency bias"""
        if len(self.recent_values) == 0:
            return self.value
        return np.mean(self.recent_values[-5:])


class GenerationAwareDiscount:
    """
    IMPROVEMENT 19: Variable discount factor based on game phase.
    Integrated into MCTS for better planning.
    """
    def __init__(self):
        self.schedules = [
            (5, 0.995),   # Early game: more far-sighted
            (10, 0.99),   # Mid game
            (14, 0.95),   # Late game: more myopic to reduce variance
        ]
    
    def get_gamma(self, generation):
        """Get discount factor for current generation"""
        for max_gen, gamma in self.schedules:
            if generation <= max_gen:
                return gamma
        return 0.95


class MCTS:
    """
    Enhanced MCTS compatible with muzero.py improvements.
    """
    def __init__(
        self,
        model,
        action_dim,
        sims=64,
        c_puct=1.25,
        discount=0.997,
        dirichlet_alpha=0.3,
        dirichlet_frac=0.25,
        generation=7,  # NEW: for adaptive discount
    ):
        self.model = model
        self.action_dim = action_dim
        self.sims = sims
        self.c_puct = c_puct
        self.discount = discount  # Base discount (can be overridden)
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac
        self.generation = generation
        
        # IMPROVEMENT 19: Adaptive discount
        self.gamma_scheduler = GenerationAwareDiscount()
        
        # Search statistics
        self.search_stats = {
            'avg_depth': 0.0,
            'max_depth': 0,
            'total_nodes': 0,
            'avg_branching_factor': 0.0,
        }
        
        # Recency bias weight (how much to favor recent good moves)
        self.recency_weight = 0.1

    def run(
        self, 
        obs, 
        legal_actions=None, 
        temperature=1.0, 
        training=True,
        head_masks=None, 
        cond_masks_obj_by_type=None,
        cond_masks_pay_by_obj=None, 
        cond_masks_extra_by_pay=None,
        factored_legal=None,
        generation=None,  # NEW: for multi-critic and adaptive discount
        terraform_pct=None,  # NEW: for multi-critic
    ):
        """
        Enhanced MCTS with support for muzero improvements.
        
        Parameters
        ----------
        obs : torch.Tensor [obs_dim]
        legal_actions : list[int] - flat slot indices
        temperature : float - controls exploration
        training : bool - add Dirichlet noise if True
        head_masks : list of 4 np.ndarray - per-head legal action masks
        cond_masks_* : dicts - conditional masks for hierarchical selection
        factored_legal : dict - mapping slot -> (t, o, p, e)
        generation : int - current game generation (for multi-critic)
        terraform_pct : float - terraform completion % (for multi-critic)
        
        Returns
        -------
        policy : np.ndarray [action_dim] - MCTS visit count distribution
        """
        if legal_actions is None or len(legal_actions) == 0:
            return np.ones(self.action_dim) / self.action_dim

        # Update generation for adaptive discount
        if generation is not None:
            self.generation = generation
        
        # Provide safe defaults for backward compatibility
        if head_masks is None:
            head_masks = [np.ones(d, dtype=np.float32) for d in FACTORED_ACTION_DIMS]
        if cond_masks_obj_by_type is None:
            cond_masks_obj_by_type = {}
        if cond_masks_pay_by_obj is None:
            cond_masks_pay_by_obj = {}
        if cond_masks_extra_by_pay is None:
            cond_masks_extra_by_pay = {}
        if factored_legal is None:
            factored_legal = {s: (0, s % OBJECT_ID_DIM, 0, 0) for s in legal_actions}
        
        # Set defaults for multi-critic
        if generation is None:
            generation = 7
        if terraform_pct is None:
            terraform_pct = 0.5

        # Initial inference with multi-critic support
        with torch.no_grad():
            # Check if model supports multi-critic (muzero.py)
            result = self.model.initial_inference(
                obs.unsqueeze(0).to(DEVICE),
                generation=generation,
                terraform_pct=terraform_pct,
            )
            
            # Unpack based on return signature
            if len(result) == 6:
                # muzero.py: (policy_heads, value, phase_values, aux_preds, h, dueling)
                policy_heads, value, phase_values, aux_preds, hidden, dueling_outputs = result
            elif len(result) == 3:
                # Original muzero.py: (policy_heads, value, h)
                policy_heads, value, hidden = result
            else:
                raise ValueError(f"Unexpected model output: {len(result)} elements")

        # Convert policy heads to numpy and apply masks
        policies = []
        for h_idx, raw in enumerate(policy_heads):
            p = torch.softmax(raw, dim=-1)[0].cpu().numpy()
            mask = head_masks[h_idx]
            p = p * mask
            s = p.sum()
            if s < 1e-8:
                p = mask / max(mask.sum(), 1.0)
            else:
                p = p / s
            policies.append(p)

        value = value.item() if torch.is_tensor(value) else float(value)

        # Create root node
        root = Node(0.0)
        root.hidden_state = hidden
        root.value_sum = value
        root.visit_count = 1

        # Expand root
        self._expand(root, policies, legal_actions, factored_legal)

        # Add exploration noise during training
        if training:
            self._add_dirichlet_noise(root)

        # Run MCTS simulations
        depths = []
        branching_factors = []
        
        for sim in range(self.sims):
            depth, branching = self._simulate(
                root, legal_actions, policies,
                head_masks, cond_masks_obj_by_type,
                cond_masks_pay_by_obj, cond_masks_extra_by_pay,
                factored_legal,
                generation, terraform_pct
            )
            depths.append(depth)
            if branching > 0:
                branching_factors.append(branching)

        # Update statistics
        self.search_stats['avg_depth'] = np.mean(depths)
        self.search_stats['max_depth'] = max(depths)
        self.search_stats['total_nodes'] = len(root.children)
        if branching_factors:
            self.search_stats['avg_branching_factor'] = np.mean(branching_factors)

        # Build visit-count policy
        visits = np.zeros(self.action_dim, dtype=np.float32)
        for a, child in root.children.items():
            visits[a] = child.visit_count

        # Temperature-based action selection
        if temperature < 0.01:
            # Greedy (deterministic)
            action = np.argmax(visits)
            pi = np.zeros_like(visits)
            pi[action] = 1.0
            return pi

        # Apply temperature
        visits = visits ** (1.0 / temperature)
        total = visits.sum()
        
        if total < 1e-8:
            # Fallback to uniform
            pi = np.zeros(self.action_dim)
            pi[legal_actions] = 1.0 / len(legal_actions)
            return pi
        
        return visits / total

    def _simulate(
        self, 
        root, 
        legal_actions, 
        policies,
        head_masks, 
        cond_masks_obj_by_type,
        cond_masks_pay_by_obj, 
        cond_masks_extra_by_pay,
        factored_legal,
        generation,
        terraform_pct
    ):
        """
        Run one MCTS simulation with adaptive discount.
        
        Returns
        -------
        depth : int
        branching_factor : float - average children per node in path
        """
        node = root
        path = [node]
        actions = []
        depth = 0
        total_children = 0

        # Selection phase
        while node.children and depth < 20:
            action, node = self._select(node)
            if node is None:
                return 0, 0
            path.append(node)
            actions.append(action)
            depth += 1
            total_children += len(node.children) if node.children else 0

        if depth >= 20:
            self._backpropagate(path, node.value, generation)
            return depth, total_children / max(depth, 1)

        # Expansion phase
        if depth > 0:
            parent = path[-2]
            action = actions[-1]

            # Build factored action one-hot
            factored_onehot = np.zeros(sum(FACTORED_ACTION_DIMS), dtype=np.float32)
            encoding = factored_legal.get(action)
            
            if encoding is not None:
                t, o, p, e = encoding
                offsets = [0]
                for d in FACTORED_ACTION_DIMS[:-1]:
                    offsets.append(offsets[-1] + d)
                factored_onehot[offsets[0] + t] = 1.0
                factored_onehot[offsets[1] + o] = 1.0
                factored_onehot[offsets[2] + p] = 1.0
                factored_onehot[offsets[3] + e] = 1.0

            action_tensor = torch.tensor(
                factored_onehot, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)

            # Recurrent inference
            with torch.no_grad():
                result = self.model.recurrent_inference(
                    parent.hidden_state, 
                    action_tensor,
                    generation=generation,
                    terraform_pct=terraform_pct,
                )
                
                # Unpack based on signature
                if len(result) == 7:
                    # muzero: (policy_heads, value, phase_values, aux, reward, h, dueling)
                    policy_heads, value, phase_values, aux, reward, hidden, dueling = result
                elif len(result) == 4:
                    # Original: (policy_heads, value, reward, h)
                    policy_heads, value, reward, hidden = result
                else:
                    raise ValueError(f"Unexpected recurrent output: {len(result)} elements")

            node.hidden_state = hidden
            node.reward = reward.item() if torch.is_tensor(reward) else float(reward)

            # Normalize policies
            rec_policies = []
            for h_idx, raw in enumerate(policy_heads):
                p = torch.softmax(raw, dim=-1)[0].cpu().numpy()
                mask = head_masks[h_idx]
                p = p * mask
                s = p.sum()
                rec_policies.append(p / s if s > 1e-8 else mask / max(mask.sum(), 1.0))

            value = value.item() if torch.is_tensor(value) else float(value)

            # Expand children
            self._expand(node, rec_policies, legal_actions, factored_legal)

        # Backpropagation with generation-aware discount
        self._backpropagate(path, value if depth > 0 else node.value, generation)
        
        avg_branching = total_children / max(depth, 1)
        return depth, avg_branching

    def _select(self, node):
        """
        IMPROVED: UCB with recency bias.
        Recent good moves get a small bonus.
        """
        best_score = -1e9
        best_action = None
        best_child = None

        sqrt_visits = math.sqrt(node.visit_count + 1)
        
        # Progressive widening
        c_puct = self.c_puct + 0.1 * math.log(node.visit_count + 1)

        for action, child in node.children.items():
            # Standard PUCT
            q_value = child.value
            
            # IMPROVEMENT: Recency bias - favor recent good performance
            recency_bonus = 0.0
            if len(child.recent_values) > 0:
                recent_avg = child.recent_value
                overall_avg = child.value
                if recent_avg > overall_avg:
                    recency_bonus = self.recency_weight * (recent_avg - overall_avg)
            
            # Exploration bonus for unvisited
            exploration_bonus = 0.1 if child.visit_count == 0 else 0.0
            
            # Combined UCB score
            u = (
                q_value +
                c_puct * child.prior * sqrt_visits / (1 + child.visit_count) +
                exploration_bonus +
                recency_bonus
            )
            
            if u > best_score:
                best_score = u
                best_action = action
                best_child = child

        return best_action, best_child

    def _expand(self, node, policies, legal_actions, factored_legal):
        """
        Expand node with factored action priors.
        Same as original but with minor optimizations.
        """
        if legal_actions is None or len(legal_actions) == 0:
            return
        
        if factored_legal is None or len(factored_legal) == 0:
            uniform_prior = 1.0 / len(legal_actions)
            for a in legal_actions:
                if a not in node.children:
                    node.children[a] = Node(uniform_prior)
            return
        
        # Compute priors as product of head probabilities
        slot_priors = {}
        for slot in legal_actions:
            if slot not in factored_legal:
                continue
            
            t, o, p, e = factored_legal[slot]
            
            # Product of factored probabilities
            prior = (
                policies[0][t] *
                policies[1][o] *
                policies[2][p] *
                policies[3][e]
            )
            slot_priors[slot] = prior
        
        # Normalize
        total = sum(slot_priors.values())
        if total < 1e-8:
            uniform_prior = 1.0 / len(legal_actions)
            for a in legal_actions:
                if a not in node.children:
                    node.children[a] = Node(uniform_prior)
        else:
            for slot, prior in slot_priors.items():
                if slot not in node.children:
                    node.children[slot] = Node(prior / total)

    def _backpropagate(self, path, leaf_value, generation=7):
        """
        IMPROVED: Backpropagation with generation-aware discount (Improvement 19).
        """
        # Get generation-specific discount
        gamma = self.gamma_scheduler.get_gamma(generation)
        
        value = leaf_value
        
        for i, node in enumerate(reversed(path)):
            # Update value sum and visit count
            node.value_sum += value
            node.visit_count += 1
            
            # Track recent values for recency bias
            node.recent_values.append(value)
            if len(node.recent_values) > 5:
                node.recent_values.pop(0)
            
            # Discount with generation-aware gamma
            value = node.reward + gamma * value
            
            # Clip for stability
            value = max(-10.0, min(10.0, value))

    def _add_dirichlet_noise(self, root):
        """
        IMPROVED: Adaptive Dirichlet noise.
        More noise when fewer legal actions (explore the few options well).
        """
        actions = list(root.children.keys())
        if len(actions) == 0:
            return
        
        # Scale alpha inversely with number of legal actions
        scaled_alpha = self.dirichlet_alpha * (self.action_dim / len(actions)) ** 0.5
        scaled_alpha = max(0.1, min(1.0, scaled_alpha))
        
        noise = np.random.dirichlet([scaled_alpha] * len(actions))
        
        for a, n in zip(actions, noise):
            root.children[a].prior = (
                root.children[a].prior * (1 - self.dirichlet_frac) +
                n * self.dirichlet_frac
            )
            root.children[a].prior = max(1e-8, root.children[a].prior)
    
    def get_stats(self):
        """Get current search statistics"""
        return self.search_stats.copy()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_mcts_with_model(
    model, 
    obs, 
    legal_actions, 
    generation=7, 
    terraform_pct=0.5,
    temperature=1.0,
    sims=100
):
    """
    Convenience function to run MCTS with muzero model.
    
    Parameters
    ----------
    model : MuZeroNetFinal
    obs : torch.Tensor [obs_dim]
    legal_actions : list[int]
    generation : int
    terraform_pct : float
    temperature : float
    sims : int
    
    Returns
    -------
    policy : np.ndarray [action_dim]
    stats : dict
    """
    mcts = MCTS(
        model=model,
        action_dim=64,  # Adjust as needed
        sims=sims,
        c_puct=1.5,
        discount=0.997,
        generation=generation
    )
    
    policy = mcts.run(
        obs=obs,
        legal_actions=legal_actions,
        temperature=temperature,
        training=True,
        generation=generation,
        terraform_pct=terraform_pct
    )
    
    return policy, mcts.get_stats()


if __name__ == "__main__":
    print("Enhanced MCTS for muzero.py")
    print("=" * 60)
    print("Features:")
    print("  ✓ Compatible with multi-critic")
    print("  ✓ Generation-aware discount (Improvement 19)")
    print("  ✓ Recency bias in UCB")
    print("  ✓ Adaptive Dirichlet noise")
    print("  ✓ Enhanced statistics tracking")
    print("=" * 60)
    
    # Test compatibility
    try:
        from muzero import MuZeroNetFinal
        model = MuZeroNetFinal(512, 64, 4096)
        print("\n✓ Compatible with muzero.py")
        
        # Test inference
        obs = torch.randn(512)
        legal_actions = [0, 1, 2, 5, 10, 15]
        
        policy, stats = run_mcts_with_model(
            model, obs, legal_actions,
            generation=7, terraform_pct=0.5,
            sims=10
        )
        
        print(f"✓ MCTS run successful")
        print(f"  Policy shape: {policy.shape}")
        print(f"  Search depth: {stats['avg_depth']:.1f}")
        print(f"  Branching factor: {stats['avg_branching_factor']:.1f}")
        
    except ImportError:
        print("\n⚠ muzero.py not found (expected in testing)")
    
    print("\n✓ MCTS module ready!")