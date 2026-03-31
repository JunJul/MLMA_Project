import torch
import torch.nn as nn

class CheXpertLoss(nn.Module):
    """
    Custom Loss for CheXpert to handle uncertain labels (-1).
    Expects targets to contain 0 (Negative), 1 (Positive), and -1 (Uncertain).
    NaNs from the CSV should be filled with 0 in the Dataset class before reaching here.
    """
    def __init__(self, policy="U-Ones", smoothing_value=0.85):
        super(CheXpertLoss, self).__init__()
        self.policy = policy
        self.smoothing_value = smoothing_value

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        logits: [Batch, 14] - Raw, un-activated outputs from the ResNet
        targets: [Batch, 14] - Ground truth labels containing 0, 1, and -1
        """

        mapped_targets = targets.clone().float()

        if self.policy == "U-Ones":
            mapped_targets[mapped_targets == -1] = 1.0
        elif self.policy == "U-Zeroes":
            mapped_targets[mapped_targets == -1] = 0.0
        elif self.policy == "U-Smooth":
            mapped_targets[mapped_targets == -1] = self.smoothing_value
        else:
            raise ValueError(f"Unknown CheXpert policy: {self.policy}. Use 'U-Ones', 'U-Zeroes', or 'U-Smooth'.")
        
        loss = self.bce(logits, mapped_targets)
        return loss