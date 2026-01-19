import os
import json
import torch
import numpy as np
import random
from pathlib import Path

class ReplayBuffer:
    def __init__(
        self,
        capacity,
        obs_dim,
        action_dim,
        device="cpu",
        directory="replay_buffer",
        cache_size=1024,
        dtype=np.float16,
    ):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.directory = Path(directory)
        self.cache_size = cache_size
        self.dtype = dtype

        self.directory.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.directory / "meta.json"
        self.obs_path = self.directory / "obs.memmap"
        self.next_obs_path = self.directory / "next_obs.memmap"
        self.actions_path = self.directory / "actions.memmap"
        self.rewards_path = self.directory / "rewards.memmap"
        self.dones_path = self.directory / "dones.memmap"
        self.policies_path = self.directory / "policies.memmap"

        self.ptr = 0
        self.ptr = 0
        self.size = 0

        # Create or load
        if self.meta_path.exists():
            self._load()
        else:
            self._create()

        # RAM cache
        self.cache_indices = None
        self.cache_obs = None
        self.cache_next_obs = None
        self.cache_actions = None
        self.cache_rewards = None
        self.cache_dones = None
        self.cache_policies = None

    # ---------------------------
    # Disk initialization
    # ---------------------------
    def _create(self):
        self.obs = np.memmap(self.obs_path, dtype=self.dtype, mode="w+", shape=(self.capacity, self.obs_dim))
        self.next_obs = np.memmap(self.next_obs_path, dtype=self.dtype, mode="w+", shape=(self.capacity, self.obs_dim))
        self.actions = np.memmap(self.actions_path, dtype=np.int16, mode="w+", shape=(self.capacity,))
        self.rewards = np.memmap(self.rewards_path, dtype=np.float16, mode="w+", shape=(self.capacity,))
        self.dones = np.memmap(self.dones_path, dtype=np.bool_, mode="w+", shape=(self.capacity,))
        self.policies = np.memmap(self.policies_path, dtype=np.float16, mode="w+", shape=(self.capacity,self.action_dim))

        self.ptr = 0
        self.size = 0
        self._save_meta()

    def _load(self):
        with open(self.meta_path, "r") as f:
            meta = json.load(f)

        self.ptr = meta["ptr"]
        self.size = meta["size"]

        self.obs = np.memmap(self.obs_path, dtype=self.dtype, mode="r+", shape=(self.capacity, self.obs_dim))
        self.next_obs = np.memmap(self.next_obs_path, dtype=self.dtype, mode="r+", shape=(self.capacity, self.obs_dim))
        self.actions = np.memmap(self.actions_path, dtype=np.int16, mode="r+", shape=(self.capacity,))
        self.rewards = np.memmap(self.rewards_path, dtype=np.float16, mode="r+", shape=(self.capacity,))
        self.dones = np.memmap(self.dones_path, dtype=np.bool_, mode="r+", shape=(self.capacity,))
        self.policies = np.memmap(self.policies_path, dtype=np.float16, mode="r+", shape=(self.capacity,self.action_dim))

    def _save_meta(self):
        tmp = {
            "ptr": self.ptr,
            "size": self.size,
            "capacity": self.capacity,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim
        }
        with open(self.meta_path, "w") as f:
            json.dump(tmp, f)

    # ---------------------------
    # Add transition (disk write)
    # ---------------------------
    def add(self, obs, action, reward, next_obs, done, policy):
        i = self.ptr

        self.obs[i] = obs.cpu().numpy().astype(self.dtype)
        self.next_obs[i] = next_obs.cpu().numpy().astype(self.dtype)
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.policies[i] = policy.cpu().numpy().astype(self.dtype)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ---------------------------
    # Cache management
    # ---------------------------
    def _load_cache(self):
        idx = np.random.randint(0, self.size, size=self.cache_size)

        self.cache_indices = idx
        self.cache_obs = self.obs[idx]
        self.cache_next_obs = self.next_obs[idx]
        self.cache_actions = self.actions[idx]
        self.cache_rewards = self.rewards[idx]
        self.cache_dones = self.dones[idx]
        self.cache_policies = self.policies[idx]

    # ---------------------------
    # Sampling
    # ---------------------------
    def sample(self, batch_size):
        if self.cache_indices is None or random.random() < 0.1:
            self._load_cache()

        idx = np.random.randint(0, len(self.cache_indices), size=batch_size)

        obs = torch.from_numpy(self.cache_obs[idx]).float().to(self.device)
        next_obs = torch.from_numpy(self.cache_next_obs[idx]).float().to(self.device)
        actions = torch.from_numpy(self.cache_actions[idx]).to(self.device)
        rewards = torch.from_numpy(self.cache_rewards[idx]).to(self.device)
        dones = torch.from_numpy(self.cache_dones[idx]).float().to(self.device)
        policies = torch.from_numpy(self.cache_policies[idx]).float().to(self.device)
        return obs, actions, rewards, next_obs, dones, policies

    # ---------------------------
    # Persistence
    # ---------------------------
    def save(self):
        self.obs.flush()
        self.next_obs.flush()
        self.actions.flush()
        self.rewards.flush()
        self.dones.flush()
        self.policies.flush()
        self._save_meta()

    def __len__(self):
        return self.size
