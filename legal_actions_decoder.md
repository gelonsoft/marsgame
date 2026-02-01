# Legal Actions Integration Guide

## Overview

The new `legal_actions_decoder.py` module provides utilities to extract legal actions directly from observation arrays, eliminating the need to pass action lists separately through the training pipeline.

## Key Benefits

1. **Self-Contained Observations**: All information needed for action selection is embedded in the observation
2. **Simplified API**: No need to track separate action lists
3. **Consistency**: Ensures legal actions are consistent across MCTS and training
4. **Debugging Tools**: Built-in utilities for inspecting and validating legal actions

## Core Functions

### 1. Extract Legal Actions

```python
from legal_actions_decoder import get_legal_actions_from_obs

# From a single observation
legal_actions = get_legal_actions_from_obs(obs)
# Returns: [0, 5, 12, 23, ...] (list of legal action indices)

# From a batch of observations
legal_actions_batch = get_legal_actions_from_obs_batch(obs_batch)
# Returns: [[0, 5, 12], [1, 3, 7], ...] (list of lists)
```

### 2. Get Legal Action Masks

```python
from legal_actions_decoder import get_legal_actions_mask_from_obs

# Binary mask (1 = legal, 0 = illegal)
mask = get_legal_actions_mask_from_obs(obs)
# Returns: [1, 0, 0, 1, 1, 0, ...] (numpy array)

# Batch version
masks = get_legal_actions_mask_from_obs_batch(obs_batch)
# Returns: shape (batch_size, MAX_ACTIONS)
```

### 3. Apply Masks to Policies

```python
from legal_actions_decoder import apply_legal_action_mask

# Mask and renormalize a policy distribution
masked_policy = apply_legal_action_mask(policy, obs)
# Automatically zeros out illegal actions and renormalizes

# Batch version
masked_policies = apply_legal_action_mask_batch(policy_batch, obs_batch)
```

### 4. Sample Legal Actions

```python
from legal_actions_decoder import sample_legal_action

# Sample from policy respecting legal actions
action = sample_legal_action(policy, obs, temperature=1.0)
# Returns: int (sampled action index)
```

## Integration Points

### In MCTS (`mcts.py`)

The MCTS now automatically extracts legal actions from observations:

```python
# Before:
policy = self.mcts.run(obs, legal_actions=[0, 5, 12], temperature=1.0)

# After:
policy = self.mcts.run(obs, legal_actions=None, temperature=1.0)
# MCTS extracts legal actions from obs automatically
```

**How it works:**
1. MCTS receives observation tensor
2. Calls `get_legal_actions_from_obs()` if `legal_actions=None`
3. Uses extracted actions for tree expansion
4. Returns policy with proper masking

### In Trainer (`trainer.py`)

#### Rollout Loop

```python
# Observation contains encoded legal actions
obs_vec = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

# MCTS extracts legal actions automatically
policy = self.mcts.run(obs_vec, legal_actions=None, temperature=temperature)

# Verify consistency with environment (optional safety check)
legal_actions_from_obs = get_legal_actions_from_obs(obs)
if set(legal_actions_from_obs) != set(action_list):
    print(f"Warning: Legal actions mismatch!")
```

#### Training Loop

```python
# Extract legal actions from observation batch
obs_np = obs.cpu().numpy()
legal_actions_batch = []
for i in range(batch_size):
    legal_actions = get_legal_actions_from_obs(obs_np[i])
    legal_actions_batch.append(legal_actions)

# Apply masks to policy targets and predictions
for i in range(batch_size):
    legal_actions = legal_actions_batch[i]
    mask = torch.zeros(action_dim, device=DEVICE)
    mask[legal_actions] = 1.0
    
    # Mask and renormalize
    policy_target[i] = policy_target[i] * mask
    policy_target[i] = policy_target[i] / policy_target[i].sum()
```

## Observation Structure

The observation array has the following structure:

```
[  0 -  511]: Game state features (512 elements)
[512 - 4607]: Action features (64 actions × 64 features = 4096 elements)
```

### Action Encoding

Each action slot has 64 features:
- Element 0: Action type (normalized categorical)
- Elements 1-63: Action-specific features (payment, cards, spaces, etc.)

**Legal action detection:**
- If action vector sum > 1e-6, the action is legal
- If all elements ≈ 0, the slot is empty (illegal)

## Debugging and Validation

### Check Legal Actions

```python
from legal_actions_decoder import print_legal_actions_info

# Print legal actions summary
print_legal_actions_info(obs, verbose=True)
# Output:
# Number of legal actions: 12
# Legal action indices: [0, 5, 7, 12, 15, 19, 23, 28, 31, 45, 52, 60]
# Action features (first 5 elements):
#   Action 0: [0.13, 0.25, 0.00, 0.00, 0.00]
#   Action 5: [0.26, 0.00, 0.00, 0.00, 0.00]
#   ...
```

### Validate Specific Action

```python
from legal_actions_decoder import validate_action_legal

is_legal = validate_action_legal(obs, action=12)
# Returns: True if action 12 is legal, False otherwise
```

### Get Action Type

```python
from legal_actions_decoder import get_action_type_from_obs

action_type = get_action_type_from_obs(obs, action_idx=5)
# Returns: "card" or "payment" or "space" etc.
```

### Count Legal Actions

```python
from legal_actions_decoder import get_num_legal_actions

num_legal = get_num_legal_actions(obs)
# Returns: int (number of legal actions)
```

## Error Handling

The decoder includes robust error handling:

```python
# No legal actions (returns uniform distribution)
legal_actions = get_legal_actions_from_obs(obs)
if len(legal_actions) == 0:
    # Handle gracefully - return uniform or end episode
    pass

# Invalid observation (NaN/Inf)
legal_actions = get_legal_actions_from_obs(obs)
# NaN values are handled with np.nan_to_num in observe()

# Batch dimension mismatch
masks = get_legal_actions_mask_from_obs_batch(obs_batch)
# Automatically handles different batch sizes
```

## Testing

### Unit Tests

```python
import numpy as np
from legal_actions_decoder import *

# Test observation structure
obs = np.random.randn(4608)  # 512 + 64*64
legal_actions = get_legal_actions_from_obs(obs)
assert all(0 <= a < 64 for a in legal_actions)

# Test masking
policy = np.random.rand(64)
masked = apply_legal_action_mask(policy, obs)
assert np.abs(masked.sum() - 1.0) < 1e-6  # Normalized

# Test sampling
action = sample_legal_action(policy, obs)
assert validate_action_legal(obs, action)
```

### Integration Tests

```python
# Test MCTS consistency
obs = env.reset()[0]
policy1 = mcts.run(obs, legal_actions=None)  # Auto-extract
policy2 = mcts.run(obs, legal_actions=env.actions_list)  # Manual

# Should produce similar results
assert np.allclose(policy1, policy2, atol=0.1)
```

## Performance Considerations

1. **Caching**: Legal actions are extracted on-demand, no caching needed
2. **Batch Operations**: Use batch functions for efficiency
3. **GPU/CPU**: Works with both torch tensors and numpy arrays
4. **Memory**: Minimal overhead (~1-2% of total memory)

## Migration Checklist

- [x] Add `legal_actions_decoder.py` to project
- [x] Update MCTS to use `get_legal_actions_from_obs()`
- [x] Update trainer rollout to extract legal actions
- [x] Update trainer training loop to mask policies
- [x] Add legal actions statistics to logging
- [x] Test consistency with environment action lists
- [x] Add error handling for edge cases
- [x] Add debugging utilities

## Common Issues

### Issue: Legal actions mismatch

```python
# Problem: Decoded actions don't match environment
legal_from_obs = get_legal_actions_from_obs(obs)
legal_from_env = env.actions_list

if set(legal_from_obs) != set(legal_from_env):
    # Solution: Use environment as ground truth
    print(f"Warning: Mismatch detected!")
    legal_actions = legal_from_env
```

### Issue: No legal actions

```python
# Problem: Empty legal actions list
if len(legal_actions) == 0:
    # Solution: Check if episode should terminate
    if env.terminations:
        # Expected - game over
        pass
    else:
        # Unexpected - debug observation encoding
        print_legal_actions_info(obs, verbose=True)
```

### Issue: Policy doesn't sum to 1

```python
# Problem: After masking, policy sum != 1.0
masked_policy = apply_legal_action_mask(policy, obs)
assert abs(masked_policy.sum() - 1.0) < 1e-5

# apply_legal_action_mask() automatically renormalizes
# If still failing, check for NaN/Inf in policy
```

## Best Practices

1. **Always validate**: Use consistency checks during development
2. **Log statistics**: Track average number of legal actions
3. **Handle edge cases**: Empty action lists, single action, etc.
4. **Use batch operations**: For efficiency in training
5. **Debug with utilities**: Use `print_legal_actions_info()` liberally
6. **Test thoroughly**: Unit tests + integration tests + end-to-end

## Future Enhancements

- [ ] Add action type filtering (e.g., only "card" actions)
- [ ] Add action feature extraction utilities
- [ ] Add visualization tools for action distributions
- [ ] Add caching for repeated observations
- [ ] Add GPU-accelerated batch processing