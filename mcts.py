import math
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Node:
    def __init__(self, prior):
        self.visit = 0
        self.value_sum = 0
        self.prior = prior
        self.children = {}

    def value(self):
        return self.value_sum / (self.visit + 1e-6)

class MCTS:
    def __init__(self, model, action_dim, sims=32, c_puct=1.25):
        self.model = model
        self.action_dim = action_dim
        self.sims = sims
        self.c_puct = c_puct

    def run(self, obs):
        policy_logits, value, h = self.model.initial_inference(obs.unsqueeze(0))
        priors = torch.softmax(policy_logits, dim=-1)[0].detach().cpu().numpy()

        root = Node(0)
        for a in range(self.action_dim):
            root.children[a] = Node(priors[a])

        for _ in range(self.sims):
            self._simulate(root, h)

        visits = np.array([root.children[a].visit for a in range(self.action_dim)])
        return visits / visits.sum()

    def _simulate(self, node, h):
        best, best_action = -1e9, None
        for a, c in node.children.items():
            ucb = c.value() + self.c_puct * c.prior * math.sqrt(node.visit + 1) / (c.visit + 1)
            if ucb > best:
                best, best_action = ucb, a

        child = node.children[best_action]

        action_onehot = torch.zeros(self.action_dim)
        action_onehot[best_action] = 1
        actions=action_onehot.unsqueeze(0)
        actions=actions.to(DEVICE)
        policy, value, reward, h_next = self.model.recurrent_inference(h, actions)

        if not child.children:
            priors = torch.softmax(policy, dim=-1)[0].detach().cpu().numpy()
            for a in range(self.action_dim):
                child.children[a] = Node(priors[a])

        child.value_sum += value.item()
        child.visit += 1
        node.visit += 1
