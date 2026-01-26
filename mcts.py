import math
import numpy as np
import torch

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

    def run(self, obs, legal_actions=None, temperature=1.0, training=True):
        """
        obs: torch.Tensor [obs_dim]
        legal_actions: list[int] or None
        """

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

        # Simulations
        for _ in range(self.sims):
            self._simulate(root, legal_actions)

        # Build policy target
        visits = np.zeros(self.action_dim, dtype=np.float32)
        for a, child in root.children.items():
            visits[a] = child.visit_count

        if temperature == 0:
            action = np.argmax(visits)
            pi = np.zeros_like(visits)
            pi[action] = 1.0
            return pi

        visits = visits ** (1.0 / temperature)
        return visits / visits.sum()

    # ----------------- Core MCTS -----------------

    def _simulate(self, root:Node, legal_actions):
        node = root
        path = [node]
        actions = []

        # Selection
        while node.children:
            action, node = self._select(node)
            path.append(node)
            actions.append(action)

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

    def _select(self, node):
        best_score = -1e9
        best_action = None
        best_child = None

        sqrt_visits = math.sqrt(node.visit_count + 1)

        for action, child in node.children.items():
            u = (
                child.value
                + self.c_puct
                * child.prior
                * sqrt_visits
                / (1 + child.visit_count)
            )
            if u > best_score:
                best_score = u
                best_action = action
                best_child = child

        return best_action, best_child

    def _expand(self, node, policy, legal_actions):
        if legal_actions is None:
            legal_actions = range(self.action_dim)

        for a in legal_actions:
            if a not in node.children:
                node.children[a] = Node(policy[a])

    def _backpropagate(self, path, leaf_value):
        value = leaf_value
        for node in reversed(path):
            node.value_sum += value
            node.visit_count += 1
            value = node.reward + self.discount * value

    # ----------------- Utilities -----------------

    def _add_dirichlet_noise(self, root):
        actions = list(root.children.keys())
        noise = np.random.dirichlet(
            [self.dirichlet_alpha] * len(actions)
        )

        for a, n in zip(actions, noise):
            root.children[a].prior = (
                root.children[a].prior * (1 - self.dirichlet_frac)
                + n * self.dirichlet_frac
            )
