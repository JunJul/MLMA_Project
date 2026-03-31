import torch.nn as nn
import torch

class MultiModalMetaLearner(nn.Module):
    def __init__(self, num_models=3, num_classes=14, num_meta_features=4):
        super(MultiModalMetaLearner, self).__init__()
        
        input_dim = (num_models * num_classes) + num_meta_features
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Output 14 final logits for CheXpertLoss (BCEWithLogitsLoss)
            nn.Linear(64, num_classes) 
        )

    def forward(self, model_probs, metadata):
        """
        model_probs: Tensor of shape [Batch, 42] containing the concatenated predictions
        metadata: Tensor of shape [Batch, 4] containing Age, Sex, Orientation, AP/PA
        """
        x = torch.cat([model_probs, metadata], dim=1)
        
        out = self.mlp(x)
        return out