import numpy as np
import torch
from typing import List, Optional
from myconfig import MAX_ACTIONS, ONE_ACTION_ARRAY_SIZE

def get_legal_actions_from_obs(obs: np.ndarray) -> List[int]:
    """
    Extract legal actions from observation array.
    
    The observation structure is:
    - First 512 elements: game state features
    - Remaining elements: action features (MAX_ACTIONS * ONE_ACTION_ARRAY_SIZE)
    
    An action is legal if its action vector is non-zero (has been encoded).
    
    Args:
        obs: Observation array from the environment
        
    Returns:
        List of legal action indices
    """
    # Handle both numpy and torch tensors
    if isinstance(obs, torch.Tensor):
        obs = obs.cpu().numpy()
    
    # Extract action features (skip first 512 game state features)
    game_state_size = 512
    action_features = obs[game_state_size:]
    
    # Reshape to (MAX_ACTIONS, ONE_ACTION_ARRAY_SIZE)
    action_features = action_features.reshape(MAX_ACTIONS, ONE_ACTION_ARRAY_SIZE)
    
    # An action is legal if its feature vector is non-zero
    # Check if any feature in the action vector is non-zero
    legal_actions = []
    for action_idx in range(MAX_ACTIONS):
        action_vec = action_features[action_idx]
        
        # Check if action vector has any non-zero elements
        # The first element (action_vec[0]) is the action type encoding
        # If it's zero, the action slot is empty
        if np.abs(action_vec).sum() > 1e-6:  # Tolerance for floating point
            legal_actions.append(action_idx)
    
    return legal_actions


def get_legal_actions_mask_from_obs(obs: np.ndarray) -> np.ndarray:
    """
    Extract legal actions mask from observation array.
    
    Args:
        obs: Observation array from the environment
        
    Returns:
        Binary mask array of shape (MAX_ACTIONS,) where 1 = legal, 0 = illegal
    """
    legal_actions = get_legal_actions_from_obs(obs)
    
    mask = np.zeros(MAX_ACTIONS, dtype=np.float32)
    mask[legal_actions] = 1.0
    
    return mask


def get_legal_actions_from_obs_batch(obs_batch: np.ndarray) -> List[List[int]]:
    """
    Extract legal actions from a batch of observations.
    
    Args:
        obs_batch: Batch of observations of shape (batch_size, obs_dim)
        
    Returns:
        List of legal action lists for each observation in the batch
    """
    # Handle both numpy and torch tensors
    if isinstance(obs_batch, torch.Tensor):
        obs_batch = obs_batch.cpu().numpy()
    
    batch_size = obs_batch.shape[0]
    legal_actions_batch = []
    
    for i in range(batch_size):
        legal_actions = get_legal_actions_from_obs(obs_batch[i])
        legal_actions_batch.append(legal_actions)
    
    return legal_actions_batch


def get_legal_actions_mask_from_obs_batch(obs_batch: np.ndarray) -> np.ndarray:
    """
    Extract legal actions masks from a batch of observations.
    
    Args:
        obs_batch: Batch of observations of shape (batch_size, obs_dim)
        
    Returns:
        Binary mask array of shape (batch_size, MAX_ACTIONS)
    """
    # Handle both numpy and torch tensors
    if isinstance(obs_batch, torch.Tensor):
        obs_batch = obs_batch.cpu().numpy()
    
    batch_size = obs_batch.shape[0]
    masks = np.zeros((batch_size, MAX_ACTIONS), dtype=np.float32)
    
    for i in range(batch_size):
        masks[i] = get_legal_actions_mask_from_obs(obs_batch[i])
    
    return masks


def validate_action_legal(obs: np.ndarray, action: int) -> bool:
    """
    Check if a specific action is legal given an observation.
    
    Args:
        obs: Observation array
        action: Action index to validate
        
    Returns:
        True if action is legal, False otherwise
    """
    legal_actions = get_legal_actions_from_obs(obs)
    return action in legal_actions


# ==================== Utility Functions ====================

def get_num_legal_actions(obs: np.ndarray) -> int:
    """
    Get the number of legal actions for an observation.
    
    Args:
        obs: Observation array
        
    Returns:
        Number of legal actions
    """
    return len(get_legal_actions_from_obs(obs))


def apply_legal_action_mask(policy: np.ndarray, obs: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Apply legal action mask to a policy distribution.
    
    Args:
        policy: Policy distribution of shape (action_dim,)
        obs: Observation array
        epsilon: Small value to avoid zeros
        
    Returns:
        Masked and renormalized policy distribution
    """
    # Get legal actions mask
    mask = get_legal_actions_mask_from_obs(obs)
    
    # Apply mask
    masked_policy = policy * mask
    
    # Add epsilon and renormalize
    masked_policy = masked_policy + epsilon
    total = masked_policy.sum()
    
    if total > epsilon:
        masked_policy = masked_policy / total
    else:
        # If all actions have zero probability, use uniform over legal actions
        legal_actions = get_legal_actions_from_obs(obs)
        if len(legal_actions) > 0:
            masked_policy = mask / len(legal_actions)
        else:
            # No legal actions - should not happen, but handle gracefully
            masked_policy = np.ones(len(policy)) / len(policy)
    
    return masked_policy


def apply_legal_action_mask_batch(policy_batch: np.ndarray, obs_batch: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Apply legal action mask to a batch of policy distributions.
    
    Args:
        policy_batch: Policy distributions of shape (batch_size, action_dim)
        obs_batch: Observation batch of shape (batch_size, obs_dim)
        epsilon: Small value to avoid zeros
        
    Returns:
        Masked and renormalized policy distributions
    """
    batch_size = policy_batch.shape[0]
    masked_policies = np.zeros_like(policy_batch)
    
    for i in range(batch_size):
        masked_policies[i] = apply_legal_action_mask(policy_batch[i], obs_batch[i], epsilon)
    
    return masked_policies


def sample_legal_action(policy: np.ndarray, obs: np.ndarray, temperature: float = 1.0) -> int:
    """
    Sample a legal action from a policy distribution.
    
    Args:
        policy: Policy distribution of shape (action_dim,)
        obs: Observation array
        temperature: Temperature for sampling (1.0 = no change, <1.0 = more greedy, >1.0 = more random)
        
    Returns:
        Sampled action index
    """
    # Apply legal action mask
    masked_policy = apply_legal_action_mask(policy, obs)
    
    # Apply temperature
    if temperature != 1.0 and temperature > 0:
        masked_policy = masked_policy ** (1.0 / temperature)
        masked_policy = masked_policy / masked_policy.sum()
    
    # Sample
    legal_actions = get_legal_actions_from_obs(obs)
    if len(legal_actions) == 0:
        raise ValueError("No legal actions available")
    
    try:
        action = np.random.choice(len(policy), p=masked_policy)
    except:
        # If sampling fails, choose uniformly from legal actions
        action = np.random.choice(legal_actions)
    
    return action


# ==================== Testing/Debugging Functions ====================

def print_legal_actions_info(obs: np.ndarray, verbose: bool = False):
    """
    Print information about legal actions in an observation.
    
    Args:
        obs: Observation array
        verbose: If True, print detailed action features
    """
    legal_actions = get_legal_actions_from_obs(obs)
    
    print(f"Number of legal actions: {len(legal_actions)}")
    print(f"Legal action indices: {legal_actions}")
    
    if verbose and len(legal_actions) > 0:
        game_state_size = 512
        action_features = obs[game_state_size:].reshape(MAX_ACTIONS, ONE_ACTION_ARRAY_SIZE)
        
        print("\nAction features (first 5 elements):")
        for action_idx in legal_actions[:10]:  # Show first 10 legal actions
            features = action_features[action_idx][:5]
            print(f"  Action {action_idx}: {features}")


def get_action_type_from_obs(obs: np.ndarray, action_idx: int) -> str:
    """
    Decode the action type from observation for a specific action.
    
    Args:
        obs: Observation array
        action_idx: Action index
        
    Returns:
        Action type as string
    """
    game_state_size = 512
    action_features = obs[game_state_size:].reshape(MAX_ACTIONS, ONE_ACTION_ARRAY_SIZE)
    
    action_vec = action_features[action_idx]
    
    # Action type is encoded in the first element
    action_type_idx = int(action_vec[0] * 23)  # 23 action types
    
    action_types = [
        "and", "or", "initialCards", "amount", "card",
        "colony", "delegate", "option", "party", "payment",
        "player", "productionToLose", "projectCard", "space",
        "aresGlobalParameters", "globalEvent", "policy",
        "resource", "resources", "xcard_choose", "xconfirm_card_choose",
        "xremove_first_card_choose", "xpayment"
    ]
    
    if action_type_idx < len(action_types):
        return action_types[action_type_idx]
    else:
        return "unknown"