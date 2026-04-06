import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # relu2_2 only — shallow, cheap, captures edges and texture
        self.features = nn.Sequential(*list(vgg.features)[:9])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x_hat, x):
        return F.mse_loss(self.features(x_hat), self.features(x))