"""
MERGED REPLAY BUFFER - COMPLETE IMPLEMENTATION
================================================

Combines:
1. ReplayBuffer - Base buffer with disk storage and caching
2. PrioritizedReplayBuffer - Enhanced buffer with prioritization and metadata

Both share the same disk storage, caching, and persistence mechanisms.
"""

import os
import json
import torch
import numpy as np
import random
from pathlib import Path
from collections import defaultdict


# ============================================================================
# BASE REPLAY BUFFER - Original with disk persistence and caching
# ============================================================================
class ReplayBuffer:
    """
    Original replay buffer with disk-backed storage and RAM caching.
    Stores: obs, action, reward, next_obs, done, policy
    """
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

        # File paths
        self.meta_path = self.directory / "meta.json"
        self.obs_path = self.directory / "obs.memmap"
        self.next_obs_path = self.directory / "next_obs.memmap"
        self.actions_path = self.directory / "actions.memmap"
        self.rewards_path = self.directory / "rewards.memmap"
        self.dones_path = self.directory / "dones.memmap"
        self.policies_path = self.directory / "policies.memmap"

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

    def _create(self):
        """Create new memory-mapped files"""
        self.obs = np.memmap(
            self.obs_path, dtype=self.dtype, mode="w+", 
            shape=(self.capacity, self.obs_dim)
        )
        self.next_obs = np.memmap(
            self.next_obs_path, dtype=self.dtype, mode="w+", 
            shape=(self.capacity, self.obs_dim)
        )
        self.actions = np.memmap(
            self.actions_path, dtype=np.int16, mode="w+", 
            shape=(self.capacity,)
        )
        self.rewards = np.memmap(
            self.rewards_path, dtype=np.float16, mode="w+", 
            shape=(self.capacity,)
        )
        self.dones = np.memmap(
            self.dones_path, dtype=np.bool_, mode="w+", 
            shape=(self.capacity,)
        )
        self.policies = np.memmap(
            self.policies_path, dtype=np.float16, mode="w+", 
            shape=(self.capacity, self.action_dim)
        )

        self.ptr = 0
        self.size = 0
        self._save_meta()

    def _load(self):
        """Load existing memory-mapped files"""
        with open(self.meta_path, "r") as f:
            meta = json.load(f)

        self.ptr = meta["ptr"]
        self.size = meta["size"]

        self.obs = np.memmap(
            self.obs_path, dtype=self.dtype, mode="r+", 
            shape=(self.capacity, self.obs_dim)
        )
        self.next_obs = np.memmap(
            self.next_obs_path, dtype=self.dtype, mode="r+", 
            shape=(self.capacity, self.obs_dim)
        )
        self.actions = np.memmap(
            self.actions_path, dtype=np.int16, mode="r+", 
            shape=(self.capacity,)
        )
        self.rewards = np.memmap(
            self.rewards_path, dtype=np.float16, mode="r+", 
            shape=(self.capacity,)
        )
        self.dones = np.memmap(
            self.dones_path, dtype=np.bool_, mode="r+", 
            shape=(self.capacity,)
        )
        self.policies = np.memmap(
            self.policies_path, dtype=np.float16, mode="r+", 
            shape=(self.capacity, self.action_dim)
        )

    def _save_meta(self):
        """Save metadata to disk"""
        meta = {
            "ptr": int(self.ptr),
            "size": int(self.size),
            "capacity": int(self.capacity),
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim)
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f)

    def add(self, obs, action, reward, next_obs, done, policy):
        """Add transition to buffer"""
        i = self.ptr

        self.obs[i] = obs.cpu().numpy().astype(self.dtype)
        self.next_obs[i] = next_obs.cpu().numpy().astype(self.dtype)
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.policies[i] = policy.cpu().numpy().astype(self.dtype)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _load_cache(self):
        """Load random subset into RAM cache"""
        if self.size == 0:
            return
        
        idx = np.random.randint(0, self.size, size=min(self.cache_size, self.size))

        self.cache_indices = idx
        self.cache_obs = self.obs[idx]
        self.cache_next_obs = self.next_obs[idx]
        self.cache_actions = self.actions[idx]
        self.cache_rewards = self.rewards[idx]
        self.cache_dones = self.dones[idx]
        self.cache_policies = self.policies[idx]

    def sample(self, batch_size):
        """Sample batch from buffer"""
        if self.size < batch_size:
            return None
        
        # Reload cache occasionally
        if self.cache_indices is None or random.random() < 0.1:
            self._load_cache()

        if self.cache_indices is None or len(self.cache_indices) == 0:
            return None

        idx = np.random.randint(0, len(self.cache_indices), size=batch_size)

        obs = torch.from_numpy(self.cache_obs[idx]).float().to(self.device)
        next_obs = torch.from_numpy(self.cache_next_obs[idx]).float().to(self.device)
        actions = torch.from_numpy(self.cache_actions[idx]).to(self.device)
        rewards = torch.from_numpy(self.cache_rewards[idx]).to(self.device)
        dones = torch.from_numpy(self.cache_dones[idx]).float().to(self.device)
        policies = torch.from_numpy(self.cache_policies[idx]).float().to(self.device)
        
        return obs, actions, rewards, next_obs, dones, policies

    def save(self):
        """Flush all data to disk"""
        self.obs.flush()
        self.next_obs.flush()
        self.actions.flush()
        self.rewards.flush()
        self.dones.flush()
        self.policies.flush()
        self._save_meta()

    def __len__(self):
        return self.size


# ============================================================================
# PRIORITIZED REPLAY BUFFER - With full metadata and disk persistence
# ============================================================================
class PrioritizedReplayBuffer:
    """
    Enhanced replay buffer with:
    - Generation-aware prioritization
    - Full metadata storage (for all 20 improvements)
    - Disk-backed storage (same as ReplayBuffer)
    - RAM caching (same as ReplayBuffer)
    - Auxiliary targets for self-supervised learning
    
    Compatible with trainer_final.py and rollout_fixed.py
    """
    
    def __init__(
        self,
        capacity,
        obs_dim,
        action_dim,
        device="cpu",
        directory="replay_prioritized",
        cache_size=2048,
        dtype=np.float16,
        priority_alpha=0.6,
        beta_init=0.4,
        beta_increment=1e-6
    ):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.directory = Path(directory)
        self.cache_size = cache_size
        self.dtype = dtype
        self.priority_alpha = priority_alpha
        self.beta = beta_init
        self.beta_increment = beta_increment

        self.directory.mkdir(parents=True, exist_ok=True)

        # File paths
        self.meta_path = self.directory / "meta.json"
        
        # Core data (same as ReplayBuffer)
        self.obs_path = self.directory / "obs.memmap"
        self.next_obs_path = self.directory / "next_obs.memmap"
        self.actions_path = self.directory / "actions.memmap"
        self.rewards_path = self.directory / "rewards.memmap"
        self.dones_path = self.directory / "dones.memmap"
        self.policies_path = self.directory / "policies.memmap"
        
        # Metadata paths (new)
        self.generations_path = self.directory / "generations.memmap"
        self.terraform_pcts_path = self.directory / "terraform_pcts.memmap"
        self.td_errors_path = self.directory / "td_errors.memmap"
        
        # Auxiliary targets paths
        self.aux_tr_delta_path = self.directory / "aux_tr_delta.memmap"
        self.aux_production_path = self.directory / "aux_production.memmap"
        self.aux_vp_delta_path = self.directory / "aux_vp_delta.memmap"
        self.aux_can_play_path = self.directory / "aux_can_play.memmap"
        self.aux_resource_suff_path = self.directory / "aux_resource_suff.memmap"

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
        self.cache_metadata = None

    def _create(self):
        """Create new memory-mapped files with metadata"""
        # Core data
        self.obs = np.memmap(
            self.obs_path, dtype=self.dtype, mode="w+", 
            shape=(self.capacity, self.obs_dim)
        )
        self.next_obs = np.memmap(
            self.next_obs_path, dtype=self.dtype, mode="w+", 
            shape=(self.capacity, self.obs_dim)
        )
        self.actions = np.memmap(
            self.actions_path, dtype=np.int16, mode="w+", 
            shape=(self.capacity,)
        )
        self.rewards = np.memmap(
            self.rewards_path, dtype=np.float16, mode="w+", 
            shape=(self.capacity,)
        )
        self.dones = np.memmap(
            self.dones_path, dtype=np.bool_, mode="w+", 
            shape=(self.capacity,)
        )
        self.policies = np.memmap(
            self.policies_path, dtype=np.float16, mode="w+", 
            shape=(self.capacity, self.action_dim)
        )
        
        # Metadata
        self.generations = np.memmap(
            self.generations_path, dtype=np.int8, mode="w+", 
            shape=(self.capacity,)
        )
        self.terraform_pcts = np.memmap(
            self.terraform_pcts_path, dtype=np.float16, mode="w+", 
            shape=(self.capacity,)
        )
        self.td_errors = np.memmap(
            self.td_errors_path, dtype=np.float16, mode="w+", 
            shape=(self.capacity,)
        )
        
        # Auxiliary targets
        self.aux_tr_delta = np.memmap(
            self.aux_tr_delta_path, dtype=np.float16, mode="w+", 
            shape=(self.capacity,)
        )
        self.aux_production = np.memmap(
            self.aux_production_path, dtype=np.float16, mode="w+", 
            shape=(self.capacity, 6)
        )
        self.aux_vp_delta = np.memmap(
            self.aux_vp_delta_path, dtype=np.float16, mode="w+", 
            shape=(self.capacity,)
        )
        self.aux_can_play = np.memmap(
            self.aux_can_play_path, dtype=np.float16, mode="w+", 
            shape=(self.capacity,)
        )
        self.aux_resource_suff = np.memmap(
            self.aux_resource_suff_path, dtype=np.float16, mode="w+", 
            shape=(self.capacity,)
        )
        
        # Initialize TD errors to 1.0
        self.td_errors[:] = 1.0

        self.ptr = 0
        self.size = 0
        self._save_meta()

    def _load(self):
        """Load existing memory-mapped files"""
        with open(self.meta_path, "r") as f:
            meta = json.load(f)

        self.ptr = meta.get("ptr", 0)
        self.size = meta.get("size", 0)

        # Core data
        self.obs = np.memmap(
            self.obs_path, dtype=self.dtype, mode="r+", 
            shape=(self.capacity, self.obs_dim)
        )
        self.next_obs = np.memmap(
            self.next_obs_path, dtype=self.dtype, mode="r+", 
            shape=(self.capacity, self.obs_dim)
        )
        self.actions = np.memmap(
            self.actions_path, dtype=np.int16, mode="r+", 
            shape=(self.capacity,)
        )
        self.rewards = np.memmap(
            self.rewards_path, dtype=np.float16, mode="r+", 
            shape=(self.capacity,)
        )
        self.dones = np.memmap(
            self.dones_path, dtype=np.bool_, mode="r+", 
            shape=(self.capacity,)
        )
        self.policies = np.memmap(
            self.policies_path, dtype=np.float16, mode="r+", 
            shape=(self.capacity, self.action_dim)
        )
        
        # Metadata
        self.generations = np.memmap(
            self.generations_path, dtype=np.int8, mode="r+", 
            shape=(self.capacity,)
        )
        self.terraform_pcts = np.memmap(
            self.terraform_pcts_path, dtype=np.float16, mode="r+", 
            shape=(self.capacity,)
        )
        self.td_errors = np.memmap(
            self.td_errors_path, dtype=np.float16, mode="r+", 
            shape=(self.capacity,)
        )
        
        # Auxiliary targets
        self.aux_tr_delta = np.memmap(
            self.aux_tr_delta_path, dtype=np.float16, mode="r+", 
            shape=(self.capacity,)
        )
        self.aux_production = np.memmap(
            self.aux_production_path, dtype=np.float16, mode="r+", 
            shape=(self.capacity, 6)
        )
        self.aux_vp_delta = np.memmap(
            self.aux_vp_delta_path, dtype=np.float16, mode="r+", 
            shape=(self.capacity,)
        )
        self.aux_can_play = np.memmap(
            self.aux_can_play_path, dtype=np.float16, mode="r+", 
            shape=(self.capacity,)
        )
        self.aux_resource_suff = np.memmap(
            self.aux_resource_suff_path, dtype=np.float16, mode="r+", 
            shape=(self.capacity,)
        )

    def _save_meta(self):
        """Save metadata to disk"""
        meta = {
            "ptr": int(self.ptr),
            "size": int(self.size),
            "capacity": int(self.capacity),
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "priority_alpha": float(self.priority_alpha),
            "beta": float(self.beta)
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f)

    def add(
        self, 
        obs, 
        action, 
        reward, 
        next_obs, 
        done, 
        policy,
        generation=1,
        terraform_pct=0.0,
        td_error=1.0,
        aux_targets=None
    ):
        """
        Add transition with full metadata.
        
        Parameters
        ----------
        obs : torch.Tensor
        action : int
        reward : float
        next_obs : torch.Tensor
        done : bool
        policy : torch.Tensor
        generation : int - game generation (1-14)
        terraform_pct : float - terraform completion (0-1)
        td_error : float - TD error magnitude
        aux_targets : dict or None - auxiliary prediction targets
        """
        i = self.ptr

        # Core data
        self.obs[i] = obs.cpu().numpy().astype(self.dtype)
        self.next_obs[i] = next_obs.cpu().numpy().astype(self.dtype)
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.policies[i] = policy.cpu().numpy().astype(self.dtype)
        
        # Metadata
        self.generations[i] = np.clip(generation, 1, 14)
        self.terraform_pcts[i] = np.clip(terraform_pct, 0.0, 1.0)
        self.td_errors[i] = abs(td_error)
        
        # Auxiliary targets
        if aux_targets is not None:
            self.aux_tr_delta[i] = aux_targets.get('tr_delta', 0.0)
            
            production = aux_targets.get('production', np.zeros(6))
            if isinstance(production, torch.Tensor):
                production = production.cpu().numpy()
            self.aux_production[i] = production
            
            self.aux_vp_delta[i] = aux_targets.get('vp_delta', 0.0)
            self.aux_can_play[i] = aux_targets.get('can_play_card', 0.0)
            self.aux_resource_suff[i] = aux_targets.get('resource_sufficient', 0.0)
        else:
            # Defaults
            self.aux_tr_delta[i] = 0.0
            self.aux_production[i] = 0.0
            self.aux_vp_delta[i] = 0.0
            self.aux_can_play[i] = 0.0
            self.aux_resource_suff[i] = 0.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def compute_priorities(self):
        """
        Compute sampling priorities based on:
        1. Generation (late game = higher priority)
        2. TD error (high error = higher priority)
        3. Terminal states (2× boost)
        
        Returns
        -------
        priorities : np.ndarray [size]
        """
        n = self.size
        if n == 0:
            return np.ones(0, dtype=np.float32)
        
        priorities = np.ones(n, dtype=np.float32)
        
        # Generation bonus (1.0 to 2.0×)
        max_gen = max(self.generations[:n].max(), 1)
        gen_weights = self.generations[:n].astype(np.float32) / max_gen
        priorities *= (1.0 + gen_weights)
        
        # TD error priority
        td_weights = self.td_errors[:n].astype(np.float32)
        priorities *= (td_weights + 1e-6)
        
        # Terminal state boost (2×)
        terminal_boost = np.where(self.dones[:n], 2.0, 1.0)
        priorities *= terminal_boost
        
        # Apply alpha exponent
        priorities = priorities ** self.priority_alpha
        
        return priorities

    def _load_cache(self):
        """Load prioritized subset into RAM cache"""
        if self.size == 0:
            return
        
        # Compute priorities
        priorities = self.compute_priorities()
        probs = priorities / priorities.sum()
        
        # Sample indices based on priorities
        n_samples = min(self.cache_size, self.size)
        idx = np.random.choice(self.size, size=n_samples, p=probs, replace=False)

        self.cache_indices = idx
        self.cache_obs = self.obs[idx].copy()
        self.cache_next_obs = self.next_obs[idx].copy()
        self.cache_actions = self.actions[idx].copy()
        self.cache_rewards = self.rewards[idx].copy()
        self.cache_dones = self.dones[idx].copy()
        self.cache_policies = self.policies[idx].copy()
        
        # Cache metadata
        self.cache_metadata = {
            'generation': self.generations[idx].copy(),
            'terraform_pct': self.terraform_pcts[idx].copy(),
            'aux_tr_delta': self.aux_tr_delta[idx].copy(),
            'aux_production': self.aux_production[idx].copy(),
            'aux_vp_delta': self.aux_vp_delta[idx].copy(),
            'aux_can_play': self.aux_can_play[idx].copy(),
            'aux_resource_suff': self.aux_resource_suff[idx].copy(),
        }

    def sample(self, batch_size):
        """
        Sample batch with prioritization and importance weights.
        
        Returns
        -------
        obs, actions, rewards, next_obs, dones, policies, metadata, weights
        
        Where metadata is dict with:
        - generation
        - terraform_pct
        - aux_targets (dict)
        
        And weights are importance sampling weights
        """
        if self.size < batch_size:
            return None
        
        # Reload cache occasionally (10% of the time)
        if self.cache_indices is None or random.random() < 0.1:
            self._load_cache()
        
        if self.cache_indices is None or len(self.cache_indices) == 0:
            return None
        
        # Sample from cache
        cache_size = len(self.cache_indices)
        idx = np.random.randint(0, cache_size, size=min(batch_size, cache_size))
        
        # Get data
        obs = torch.from_numpy(self.cache_obs[idx]).float().to(self.device)
        next_obs = torch.from_numpy(self.cache_next_obs[idx]).float().to(self.device)
        actions = torch.from_numpy(self.cache_actions[idx]).to(self.device)
        rewards = torch.from_numpy(self.cache_rewards[idx]).to(self.device)
        dones = torch.from_numpy(self.cache_dones[idx]).float().to(self.device)
        policies = torch.from_numpy(self.cache_policies[idx]).float().to(self.device)
        
        # Get metadata
        metadata = {
            'generation': torch.from_numpy(
                self.cache_metadata['generation'][idx]
            ).to(self.device),
            'terraform_pct': torch.from_numpy(
                self.cache_metadata['terraform_pct'][idx]
            ).to(self.device),
            'aux_targets': {
                'tr_delta': torch.from_numpy(
                    self.cache_metadata['aux_tr_delta'][idx]
                ).unsqueeze(-1).float().to(self.device),
                'production': torch.from_numpy(
                    self.cache_metadata['aux_production'][idx]
                ).float().to(self.device),
                'vp_delta': torch.from_numpy(
                    self.cache_metadata['aux_vp_delta'][idx]
                ).unsqueeze(-1).float().to(self.device),
                'can_play_card': torch.from_numpy(
                    self.cache_metadata['aux_can_play'][idx]
                ).unsqueeze(-1).float().to(self.device),
                'resource_sufficient': torch.from_numpy(
                    self.cache_metadata['aux_resource_suff'][idx]
                ).unsqueeze(-1).float().to(self.device),
            }
        }
        
        # Compute importance sampling weights
        # Uniform weights for now (can be enhanced later)
        weights = torch.ones(len(idx), dtype=torch.float32, device=self.device)
        
        # Increment beta toward 1.0
        self.beta = min(1.0, self.beta + self.beta_increment * batch_size)
        
        return obs, actions, rewards, next_obs, dones, policies, metadata, weights

    def update_priorities(self, indices, td_errors):
        """
        Update priorities for given indices.
        
        Parameters
        ----------
        indices : np.ndarray - indices in the buffer
        td_errors : np.ndarray - new TD errors
        """
        for idx, td_error in zip(indices, td_errors):
            if 0 <= idx < self.size:
                self.td_errors[idx] = abs(td_error)

    def save(self):
        """Flush all data to disk"""
        # Core data
        self.obs.flush()
        self.next_obs.flush()
        self.actions.flush()
        self.rewards.flush()
        self.dones.flush()
        self.policies.flush()
        
        # Metadata
        self.generations.flush()
        self.terraform_pcts.flush()
        self.td_errors.flush()
        
        # Auxiliary targets
        self.aux_tr_delta.flush()
        self.aux_production.flush()
        self.aux_vp_delta.flush()
        self.aux_can_play.flush()
        self.aux_resource_suff.flush()
        
        self._save_meta()

    def __len__(self):
        return self.size
    
    def get_stats(self):
        """Get buffer statistics"""
        if self.size == 0:
            return {}
        
        return {
            'size': self.size,
            'capacity': self.capacity,
            'avg_generation': self.generations[:self.size].mean(),
            'avg_terraform_pct': self.terraform_pcts[:self.size].mean(),
            'avg_td_error': self.td_errors[:self.size].mean(),
            'terminal_states': self.dones[:self.size].sum(),
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def migrate_replay_buffer(old_buffer, new_buffer):
    """
    Migrate data from ReplayBuffer to PrioritizedReplayBuffer.
    
    Parameters
    ----------
    old_buffer : ReplayBuffer
    new_buffer : PrioritizedReplayBuffer
    """
    print(f"Migrating {len(old_buffer)} transitions...")
    
    for i in range(len(old_buffer)):
        obs = torch.from_numpy(old_buffer.obs[i].astype(np.float32))
        next_obs = torch.from_numpy(old_buffer.next_obs[i].astype(np.float32))
        action = int(old_buffer.actions[i])
        reward = float(old_buffer.rewards[i])
        done = bool(old_buffer.dones[i])
        policy = torch.from_numpy(old_buffer.policies[i].astype(np.float32))
        
        # Add with default metadata
        new_buffer.add(
            obs, action, reward, next_obs, done, policy,
            generation=7,  # Default
            terraform_pct=0.5,  # Default
            td_error=1.0,  # Default
            aux_targets=None
        )
        
        if (i + 1) % 1000 == 0:
            print(f"  Migrated {i + 1}/{len(old_buffer)}")
    
    print("✓ Migration complete")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING MERGED REPLAY BUFFERS")
    print("=" * 60)
    
    # Test 1: Basic ReplayBuffer
    print("\n1. Testing ReplayBuffer...")
    buffer = ReplayBuffer(
        capacity=100,
        obs_dim=512,
        action_dim=64,
        device='cpu',
        directory='test_replay_basic'
    )
    
    # Add some data
    for i in range(10):
        obs = torch.randn(512)
        next_obs = torch.randn(512)
        action = i % 64
        reward = float(i)
        done = (i % 5 == 4)
        policy = torch.randn(64)
        
        buffer.add(obs, action, reward, next_obs, done, policy)
    
    assert len(buffer) == 10
    print(f"   ✓ Added 10 transitions")
    
    # Sample
    sample = buffer.sample(5)
    assert sample is not None
    print(f"   ✓ Sampled batch of 5")
    
    # Save and reload
    buffer.save()
    buffer2 = ReplayBuffer(
        capacity=100,
        obs_dim=512,
        action_dim=64,
        directory='test_replay_basic'
    )
    assert len(buffer2) == 10
    print(f"   ✓ Saved and reloaded")
    
    # Test 2: PrioritizedReplayBuffer
    print("\n2. Testing PrioritizedReplayBuffer...")
    pri_buffer = PrioritizedReplayBuffer(
        capacity=100,
        obs_dim=512,
        action_dim=64,
        device='cpu',
        directory='test_replay_prioritized'
    )
    
    # Add with metadata
    for i in range(10):
        obs = torch.randn(512)
        next_obs = torch.randn(512)
        action = i % 64
        reward = float(i)
        done = (i % 5 == 4)
        policy = torch.randn(64)
        
        aux_targets = {
            'tr_delta': float(i % 3),
            'production': np.random.randn(6),
            'vp_delta': float(i % 2),
            'can_play_card': float(i % 2),
            'resource_sufficient': 1.0
        }
        
        pri_buffer.add(
            obs, action, reward, next_obs, done, policy,
            generation=min(i + 1, 14),
            terraform_pct=i / 10.0,
            td_error=abs(reward),
            aux_targets=aux_targets
        )
    
    assert len(pri_buffer) == 10
    print(f"   ✓ Added 10 transitions with metadata")
    
    # Sample with metadata
    result = pri_buffer.sample(5)
    assert result is not None
    obs, act, rew, next_obs, done, pol, metadata, weights = result
    
    assert 'generation' in metadata
    assert 'aux_targets' in metadata
    assert 'tr_delta' in metadata['aux_targets']
    print(f"   ✓ Sampled with full metadata")
    
    # Check stats
    stats = pri_buffer.get_stats()
    print(f"   ✓ Stats: gen={stats['avg_generation']:.1f}, "
          f"terraform={stats['avg_terraform_pct']:.2f}")
    
    # Save and reload
    pri_buffer.save()
    pri_buffer2 = PrioritizedReplayBuffer(
        capacity=100,
        obs_dim=512,
        action_dim=64,
        directory='test_replay_prioritized'
    )
    assert len(pri_buffer2) == 10
    stats2 = pri_buffer2.get_stats()
    assert abs(stats2['avg_generation'] - stats['avg_generation']) < 0.1
    print(f"   ✓ Saved and reloaded with metadata preserved")
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
    print("\nBoth replay buffers are working correctly!")
    print("- ReplayBuffer: Basic with disk storage")
    print("- PrioritizedReplayBuffer: Enhanced with metadata")