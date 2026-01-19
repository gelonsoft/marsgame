
import random
import torch
import numpy as np
from collections import deque

#Used for:
#off-policy training
#MuZero training
#value reanalysis
#prioritized replay

class ReplayBuffer:
    def __init__(self, capacity=3_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done, policy):
        self.buffer.append((obs.cpu(), action, reward, next_obs.cpu(), done, policy))

    def sample(self, batch_size):
        #batch=[]
        #for i in range(batch_size):
        #    batch.append(self.buffer.pop())
        batch = random.sample(self.buffer, batch_size)
        obs, act, rew, next_obs, done, policy = zip(*batch)

        return (
            torch.stack(obs),
            torch.tensor(act),
            torch.tensor(rew, dtype=torch.float32),
            torch.stack(next_obs),
            torch.tensor(done, dtype=torch.float32),
            torch.tensor(policy, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)
