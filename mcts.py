import math
import numpy as np
import torch
from factored_actions import FACTORED_ACTION_DIMS, ACTION_TYPE_DIM, OBJECT_ID_DIM, PAYMENT_ID_DIM, EXTRA_PARAM_DIM, NUM_HEADS

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

    def run(self, obs, legal_actions=None, temperature=1.0, training=True,
            head_masks=None, cond_masks_obj_by_type=None,
            cond_masks_pay_by_obj=None, cond_masks_extra_by_pay=None,
            factored_legal=None):
        """
        Factored MCTS: policy outputs 4 heads.  Selection descends the
        factor tree (type → object → payment → extra) using conditional
        masks at each level.

        Parameters
        ----------
        obs : torch.Tensor [obs_dim]
        legal_actions : list[int]  — flat slot indices (used for visit-count output)
        head_masks : list of 4 np.ndarray  — unconditional per-head masks
        cond_masks_* : dicts of conditional masks built by the env
        factored_legal : dict  slot -> (t, o, p, e)
        """
        if legal_actions is None or len(legal_actions) == 0:
            return np.ones(self.action_dim) / self.action_dim

        # Provide safe defaults so callers that haven't upgraded yet still work
        if head_masks is None:
            head_masks = [np.ones(d, dtype=np.float32) for d in FACTORED_ACTION_DIMS]
        if cond_masks_obj_by_type is None:
            cond_masks_obj_by_type = {}
        if cond_masks_pay_by_obj is None:
            cond_masks_pay_by_obj = {}
        if cond_masks_extra_by_pay is None:
            cond_masks_extra_by_pay = {}
        if factored_legal is None:
            # Fallback: treat each legal action as type=0, obj=slot%32, pay=0, extra=0
            factored_legal = {s: (0, s % OBJECT_ID_DIM, 0, 0) for s in legal_actions}

        with torch.no_grad():
            policy_heads, value, hidden = self.model.initial_inference(
                obs.unsqueeze(0).to(DEVICE)
            )
            # policy_heads: list of 4 tensors, each [1, head_dim]

        # Convert to numpy, apply masks, normalise each head
        policies = []
        for h_idx, raw in enumerate(policy_heads):
            p = torch.softmax(raw, dim=-1)[0].cpu().numpy()
            mask = head_masks[h_idx]
            p = p * mask
            s = p.sum()
            if s < 1e-8:
                # Uniform over legal entries in this head
                p = mask / max(mask.sum(), 1.0)
            else:
                p = p / s
            policies.append(p)

        value = value.item()

        root = Node(0.0)
        root.hidden_state = hidden
        root.value_sum = value
        root.visit_count = 1

        # Root expansion over flat legal slots (for visit-count aggregation)
        self._expand(root, policies, legal_actions, factored_legal)

        if training:
            self._add_dirichlet_noise(root)

        depths = []
        for _ in range(self.sims):
            depth = self._simulate(root, legal_actions, policies,
                                   head_masks, cond_masks_obj_by_type,
                                   cond_masks_pay_by_obj, cond_masks_extra_by_pay,
                                   factored_legal)
            depths.append(depth)

        self.search_stats['avg_depth'] = np.mean(depths)
        self.search_stats['max_depth'] = max(depths)

        # Build visit-count policy target (flat, over MAX_ACTIONS slots)
        visits = np.zeros(self.action_dim, dtype=np.float32)
        for a, child in root.children.items():
            visits[a] = child.visit_count

        if temperature < 0.01:
            action = np.argmax(visits)
            pi = np.zeros_like(visits)
            pi[action] = 1.0
            return pi

        visits = visits ** (1.0 / temperature)
        total = visits.sum()
        if total < 1e-8:
            pi = np.zeros(self.action_dim)
            pi[legal_actions] = 1.0 / len(legal_actions)
            return pi
        return visits / total

    # --------------- Factored action selection helper ----------------
    def _select_factored_action(self, policies, head_masks,
                                cond_masks_obj_by_type, cond_masks_pay_by_obj,
                                cond_masks_extra_by_pay, factored_legal):
        """
        Greedily descend the 4-level factor tree using conditional masks.
        Returns the flat slot that matches (t, o, p, e), or None.
        """
        # Head 0: action_type
        masked_p = policies[0] * head_masks[0]
        if masked_p.sum() < 1e-8:
            return None
        t = int(np.random.choice(len(masked_p), p=masked_p / masked_p.sum()))

        # Head 1: object_id  (conditioned on t)
        obj_mask = cond_masks_obj_by_type.get(t, np.zeros(OBJECT_ID_DIM, dtype=np.float32))
        masked_p = policies[1] * obj_mask
        if masked_p.sum() < 1e-8:
            return None
        o = int(np.random.choice(len(masked_p), p=masked_p / masked_p.sum()))

        # Head 2: payment_id  (conditioned on t, o)
        pay_mask = cond_masks_pay_by_obj.get((t, o), np.zeros(PAYMENT_ID_DIM, dtype=np.float32))
        masked_p = policies[2] * pay_mask
        if masked_p.sum() < 1e-8:
            return None
        p = int(np.random.choice(len(masked_p), p=masked_p / masked_p.sum()))

        # Head 3: extra  (conditioned on t, o, p)
        ext_mask = cond_masks_extra_by_pay.get((t, o, p), np.zeros(EXTRA_PARAM_DIM, dtype=np.float32))
        masked_p = policies[3] * ext_mask
        if masked_p.sum() < 1e-8:
            return None
        e = int(np.random.choice(len(masked_p), p=masked_p / masked_p.sum()))

        # Reverse-lookup: find the flat slot whose factored encoding matches
        for slot, enc in factored_legal.items():
            if enc == (t, o, p, e):
                return slot
        return None

    # ----------------- Core MCTS -----------------

    def _simulate(self, root: Node, legal_actions, policies,
                  head_masks, cond_masks_obj_by_type,
                  cond_masks_pay_by_obj, cond_masks_extra_by_pay,
                  factored_legal):
        """Returns the depth of the simulation.  Uses factored action
        selection when expanding new nodes."""
        node = root
        path = [node]
        actions = []
        depth = 0

        # Selection
        while node.children and depth < 20:
            action, node = self._select(node)
            if node is None:
                return 0
            path.append(node)
            actions.append(action)
            depth += 1

        if depth >= 20:
            self._backpropagate(path, node.value)
            return depth

        parent = path[-2]
        action = actions[-1]

        # Build factored action one-hot for dynamics network input.
        # Shape: [1, sum(FACTORED_ACTION_DIMS)]  — concatenation of 4 one-hots.
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

        action_tensor = torch.tensor(factored_onehot, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # Expand
        with torch.no_grad():
            policy_heads, value, reward, hidden = self.model.recurrent_inference(
                parent.hidden_state, action_tensor
            )

        node.hidden_state = hidden
        node.reward = reward.item()

        # Normalise each head policy
        rec_policies = []
        for h_idx, raw in enumerate(policy_heads):
            p = torch.softmax(raw, dim=-1)[0].cpu().numpy()
            mask = head_masks[h_idx]
            p = p * mask
            s = p.sum()
            rec_policies.append(p / s if s > 1e-8 else mask / max(mask.sum(), 1.0))

        value = value.item()

        # Expand children using factored policies
        self._expand(node, rec_policies, legal_actions, factored_legal)

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

    def _expand(self, node, policies, legal_actions, factored_legal):
        """
        Expand node children using factored policies.
        
        Parameters
        ----------
        node : Node
        policies : list of 4 np.ndarray — normalized policies for each head
        legal_actions : list[int] — flat slot indices that are legal
        factored_legal : dict[int, tuple] — mapping slot -> (t, o, p, e)
        
        For each legal slot, compute its prior as the product of the 4 head
        probabilities: P(slot) = P(type) * P(obj|type) * P(pay|obj) * P(extra|pay)
        """
        if legal_actions is None or len(legal_actions) == 0:
            return
        
        if factored_legal is None or len(factored_legal) == 0:
            # Fallback: uniform priors if factored data not available
            uniform_prior = 1.0 / len(legal_actions)
            for a in legal_actions:
                if a not in node.children:
                    node.children[a] = Node(uniform_prior)
            return
        
        # Compute prior for each legal slot by multiplying head probabilities
        slot_priors = {}
        for slot in legal_actions:
            if slot not in factored_legal:
                continue
            
            t, o, p, e = factored_legal[slot]
            
            # Product of head probabilities (already normalized per-head)
            prior = (
                policies[0][t] *
                policies[1][o] *
                policies[2][p] *
                policies[3][e]
            )
            slot_priors[slot] = prior
        
        # Normalize slot priors
        total = sum(slot_priors.values())
        if total < 1e-8:
            # Fallback to uniform if all priors are zero
            uniform_prior = 1.0 / len(legal_actions)
            for a in legal_actions:
                if a not in node.children:
                    node.children[a] = Node(uniform_prior)
        else:
            # Normalized priors
            for slot, prior in slot_priors.items():
                if slot not in node.children:
                    node.children[slot] = Node(prior / total)

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