import torch
import torch.nn as nn
import torch.nn.functional as F

class CheXpertLoss(nn.Module):
    """
    Custom Loss for CheXpert to handle uncertain labels (-1).
    Expects targets to contain 0 (Negative), 1 (Positive), and -1 (Uncertain).
    NaNs from the CSV should be filled with 0 in the Dataset class before reaching here.
    """
    def __init__(self, policy="U-Ones"):
        super().__init__()
        self.policy = policy

    def forward(self, logits, targets):
        # STEP 1: Clone targets so we don't accidentally overwrite the original dataset
        processed_targets = targets.clone()
        
        # STEP 2: Apply the Uncertainty Policy to the real data
        if self.policy == "U-Ones":
            processed_targets[processed_targets == -1.0] = 1.0
        elif self.policy == "U-Zeros":
            processed_targets[processed_targets == -1.0] = 0.0
        elif self.policy == "U-Smooth":
            processed_targets[processed_targets == -1.0] = 0.55 
        elif self.policy == "U-Ignore":
            # If you want to completely ignore uncertain labels, turn them into NaNs!
            processed_targets[processed_targets == -1.0] = float('nan')

        # STEP 3: Create the mask to hide all the NaNs (both synthetic and U-Ignore)
        valid_mask = ~torch.isnan(processed_targets)
        
        # STEP 4: Extract ONLY the mathematically safe logits and targets
        valid_logits = logits[valid_mask]
        valid_targets = processed_targets[valid_mask]
        
        # Safety check in case an entire batch is somehow empty
        if valid_logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        # STEP 5: Calculate the final loss
        loss = F.binary_cross_entropy_with_logits(valid_logits, valid_targets, reduction='mean')
        
        return loss