import torch
from torch.distributions.categorical import Categorical

def mask_logits(logits, action_mask):
    """
    Modify logits by masking rightmost elements based on action_mask.
    
    Args:
        logits: Input tensor of shape (batch_size, LOGITS_SIZE)
        action_mask: List of integers where each integer represents how many 
                    rightmost elements to mask in each row (0 means mask all)
    
    Returns:
        Modified logits tensor with appropriate elements masked
    """
    # Ensure inputs are on the same device
    device = logits.device
    batch_size, logits_size = logits.shape
    
    # Convert action_mask to a tensor
    action_mask = torch.tensor(action_mask, dtype=torch.long, device=device)
    
    # Create a mask tensor initialized with -inf (to effectively disable those logits)
    mask = torch.full_like(logits, float('-inf'))
    
    # For each row, keep the first (logits_size - mask_value) elements unmasked
    for i in range(batch_size):
        mask_value = action_mask[i]
        if mask_value == 0:
            # Mask all elements
            mask[i, :] = 0
        else:
            # Keep first (logits_size - mask_value) elements, mask the rest
            keep_count = logits_size - mask_value
            if keep_count > 0:
                mask[i, :keep_count] = 0
    
    # Apply the mask to the logits
    modified_logits = logits + mask
    
    return modified_logits


logits=torch.randn(2,7)
action_mask=[3,5]
modified_logits=mask_logits(logits, action_mask)
probs = Categorical(logits=modified_logits)
print(f"Original logits:\n{logits}")
print(f"Modified logits:\n{modified_logits}")
print(f"Action mask: {action_mask}")
print(f"Probs={probs}")

