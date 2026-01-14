import copy
import random
import torch

class SelfPlayPool:
    def __init__(self, max_size=5):
        self.pool = []
        self.max_size = max_size

    def add(self, model):
        snapshot = copy.deepcopy(model).cpu()
        self.pool.append(snapshot)
        if len(self.pool) > self.max_size:
            self.pool.pop(0)

    def sample(self):
        if not self.pool:
            return None
        return random.choice(self.pool)
