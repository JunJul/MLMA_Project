import torch.nn as nn
import torch

class MultiModalMetaLearner(nn.Module):
    def __init__(self, num_models=3, num_classes=14, num_meta_features=4, pca_dim=None):
        super(MultiModalMetaLearner, self).__init__()
        
        if pca_dim is not None:
            input_dim = pca_dim + num_meta_features
        else:
            input_dim = (num_models * num_classes) + num_meta_features
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output 14 final logits for CheXpertLoss (BCEWithLogitsLoss)
            nn.Linear(128, num_classes) 
        )

    def forward(self, model_probs, metadata):
        """
        model_probs: Tensor of shape [Batch, 42] containing the concatenated predictions
        metadata: Tensor of shape [Batch, 4] containing Age, Sex, Orientation, AP/PA
        """
        x = torch.cat([model_probs, metadata], dim=1)
        
        out = self.mlp(x)
        return out

# Alias so dynamic import with "models.MLP" resolves correctly
MLP = MultiModalMetaLearner