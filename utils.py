# import torch
import torch.nn as nn
import torch.nn.functional as F


def check_data_loader(data_loader):
    for _, data_point in enumerate(data_loader):
        img, label = data_point
        print(img.shape)
        print(img)
        print(label)
        break

class LabelSmoothingCorssEntropyLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(LabelSmoothingCorssEntropyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, outputs, targets):
        """
        Args:
            outputs: Tensor of shape (batch_size, num_classes)
            targets: Tensor of shape (batch_size) containing integer labels
        """

        log_probs = F.log_softmax(outputs, dim=-1)

        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)

        smooth_loss = -log_probs.mean(dim=-1)

        loss = (1 - self.alpha) * nll_loss + self.alpha * smooth_loss
        
        return loss.mean()

class EarlyStopping:
    """
    Stops training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0