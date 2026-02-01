import math
import numpy as np
import torch
from legal_actions_decoder import get_legal_actions_from_obs, apply_legal_action_mask

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Node:
    def __init__(self, prior):
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.reward = 0.0
        self.children = {}
        self.hidden_state = None

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(
        self,
        model,
        action_dim,
        sims=64,
        c_puct=1.25,
        discount=0.997,
        dirichlet_alpha=0.3,
        dirichlet_frac=0.25,
    ):
        self.model = model
        self.action_dim = action_dim
        self.sims = sims
        self.c_puct = c_puct
        self.discount = discount
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac
        
        # Keep track of search statistics
        self.search_stats = {
            'avg_depth': 0.0,
            'max_depth': 0,
            'total_nodes': 0
        }

    def run(self, obs, legal_actions=None, temperature=1.0, training=True):
        """
        IMPROVED: Extract legal actions from observation if not provided
        
        obs: torch.Tensor [obs_dim]
        legal_actions: list[int] or None (will be extracted from obs if None)
        """
        
        # NEW: Extract legal actions from observation if not provided
        if legal_actions is None:
            legal_actions = get_legal_actions_from_obs(obs.cpu().numpy() if isinstance(obs, torch.Tensor) else obs)
        
        # Handle edge case: no legal actions
        if len(legal_actions) == 0:
            # Return uniform distribution
            return np.ones(self.action_dim) / self.action_dim

        with torch.no_grad():
            policy_logits, value, hidden = self.model.initial_inference(
                obs.unsqueeze(0).to(DEVICE)
            )

        policy = torch.softmax(policy_logits, dim=-1)[0].cpu().numpy()
        value = value.item()

        root = Node(0.0)
        root.hidden_state = hidden
        root.value_sum = value
        root.visit_count = 1

        # Root expansion
        self._expand(root, policy, legal_actions)

        # Dirichlet noise (training only)
        if training:
            self._add_dirichlet_noise(root)

        # Simulations with depth tracking
        depths = []
        for _ in range(self.sims):
            depth = self._simulate(root, legal_actions)
            depths.append(depth)
        
        # Update statistics
        self.search_stats['avg_depth'] = np.mean(depths)
        self.search_stats['max_depth'] = max(depths)

        # Build policy target
        visits = np.zeros(self.action_dim, dtype=np.float32)
        for a, child in root.children.items():
            visits[a] = child.visit_count

        if temperature == 0:
            action = np.argmax(visits)
            pi = np.zeros_like(visits)
            pi[action] = 1.0
            return pi

        # Better temperature handling
        if temperature < 0.01:  # Very low temperature
            action = np.argmax(visits)
            pi = np.zeros_like(visits)
            pi[action] = 1.0
            return pi
        
        visits = visits ** (1.0 / temperature)
        total = visits.sum()
        
        if total < 1e-8:  # Safety check
            # Return uniform over legal actions
            pi = np.zeros(self.action_dim)
            pi[legal_actions] = 1.0 / len(legal_actions)
            return pi
            
        return visits / total

    # ----------------- Core MCTS -----------------

    def _simulate(self, root: Node, legal_actions):
        """Returns the depth of the simulation"""
        node = root
        path = [node]
        actions = []
        depth = 0

        # Selection
        while node.children and depth < 20:  # Max depth limit
            action, node = self._select(node)
            if node is None:
                return 0
            path.append(node)
            actions.append(action)
            depth += 1

        # Don't expand if we hit max depth
        if depth >= 20:
            self._backpropagate(path, node.value)
            return depth

        parent = path[-2]
        action = actions[-1]

        # Expand
        with torch.no_grad():
            action_onehot = torch.zeros(1, self.action_dim, device=DEVICE)
            action_onehot[0, action] = 1.0

            policy, value, reward, hidden = self.model.recurrent_inference(
                parent.hidden_state, action_onehot
            )

        node.hidden_state = hidden
        node.reward = reward.item()

        policy = torch.softmax(policy, dim=-1)[0].cpu().numpy()
        value = value.item()

        self._expand(node, policy, legal_actions)

        # Backup
        self._backpropagate(path, value)
        
        return depth

    def _select(self, node):
        """Better UCB formula with dynamic exploration"""
        best_score = -1e9
        best_action = None
        best_child = None

        sqrt_visits = math.sqrt(node.visit_count + 1)
        
        # Progressive widening - increase c_puct as node is visited more
        c_puct = self.c_puct + 0.1 * math.log(node.visit_count + 1)

        for action, child in node.children.items():
            # Standard PUCT formula
            q_value = child.value
            
            # Add small bonus for unexplored actions
            exploration_bonus = 0.0
            if child.visit_count == 0:
                exploration_bonus = 0.1
            
            u = (
                q_value
                + c_puct * child.prior * sqrt_visits / (1 + child.visit_count)
                + exploration_bonus
            )
            
            if u > best_score:
                best_score = u
                best_action = action
                best_child = child

        return best_action, best_child

    def _expand(self, node, policy, legal_actions):
        """Better handling of legal actions and prior normalization"""
        if legal_actions is None or len(legal_actions) == 0:
            legal_actions = range(self.action_dim)

        # Normalize policy only over legal actions
        legal_policy_sum = sum(policy[a] for a in legal_actions)
        
        if legal_policy_sum < 1e-8:
            # Uniform distribution if policy is too small
            uniform_prior = 1.0 / len(legal_actions)
            for a in legal_actions:
                if a not in node.children:
                    node.children[a] = Node(uniform_prior)
        else:
            # Normalize policy over legal actions
            for a in legal_actions:
                if a not in node.children:
                    normalized_prior = policy[a] / legal_policy_sum
                    node.children[a] = Node(normalized_prior)

    def _backpropagate(self, path, leaf_value):
        """Better value backup with normalization"""
        value = leaf_value
        
        for i, node in enumerate(reversed(path)):
            node.value_sum += value
            node.visit_count += 1
            
            # Discount the value
            value = node.reward + self.discount * value
            
            # Optional value clipping for stability
            value = max(-10.0, min(10.0, value))

    # ----------------- Utilities -----------------

    def _add_dirichlet_noise(self, root):
        """
        Adaptive Dirichlet noise - more exploration when fewer legal actions
        """
        actions = list(root.children.keys())
        if len(actions) == 0:
            return
        
        # Key improvement: scale alpha INVERSELY with legal actions
        # Fewer legal actions = more noise = more exploration of those few options
        # More legal actions = less noise = trust the policy more
        scaled_alpha = self.dirichlet_alpha * (self.action_dim / len(actions)) ** 0.5
        
        # Clamp alpha to reasonable range [0.1, 1.0]
        scaled_alpha = max(0.1, min(1.0, scaled_alpha))
        
        noise = np.random.dirichlet([scaled_alpha] * len(actions))
        
        for a, n in zip(actions, noise):
            root.children[a].prior = (
                root.children[a].prior * (1 - self.dirichlet_frac)
                + n * self.dirichlet_frac
            )
            # Ensure priors stay positive and meaningful
            root.children[a].prior = max(1e-8, root.children[a].prior)