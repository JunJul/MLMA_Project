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



# =========================
# Grad-CAM utilities
# Paste this block at the end of utils.py
# =========================

def _find_target_layer_for_gradcam(model):
    """
    Prefer ResNet layer4[-1]; otherwise fall back to the last Conv2d layer.
    """
    import torch.nn as nn

    if hasattr(model, "layer4"):
        try:
            return model.layer4[-1]
        except Exception:
            return model.layer4

    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m

    if last_conv is None:
        raise ValueError("Grad-CAM: cannot find a convolution layer. Please specify target_layer manually.")

    return last_conv


def _unnormalize_to_rgb_float(x_1xchw, mean, std):
    """
    Convert normalized (1,C,H,W) tensor back to (H,W,3) float in [0,1] for visualization.
    """
    import torch
    import numpy as np

    x = x_1xchw.detach().cpu()[0]  # (C,H,W)

    # If 1-channel, replicate to 3-channel
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)

    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)

    x = x * std_t + mean_t
    x = x.clamp(0, 1)

    rgb = x.permute(1, 2, 0).numpy().astype(np.float32)  # (H,W,3)
    return rgb


def gradcam_overlay(
    model,
    input_tensor_bchw,
    target_cls=None,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    target_layer=None
):
    """
    Generate a Grad-CAM heatmap overlay image.

    Args:
        model: classification model that outputs logits
        input_tensor_bchw: (B,C,H,W) normalized input tensor
        target_cls: which class to explain; None means explain predicted class
        mean/std: for unnormalization back to [0,1] RGB float
        target_layer: optionally specify layer; otherwise auto-detect

    Returns:
        cam_image: uint8 (H,W,3) heatmap overlay
        probs: numpy (num_classes,) softmax probabilities
        pred_cls: int predicted class index
    """
    import torch
    import numpy as np

    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError as e:
        raise ImportError(
            "Grad-CAM dependencies are missing. Install with: pip install grad-cam opencv-python pillow"
        ) from e

    device = next(model.parameters()).device
    model.eval()

    x = input_tensor_bchw[:1].to(device)

    # Predict (no gradients needed)
    with torch.no_grad():
        logits = model(x)
        probs_t = torch.softmax(logits, dim=1)[0]
        pred_cls = int(torch.argmax(probs_t).item())
        probs = probs_t.detach().cpu().numpy()

    if target_cls is None:
        target_cls = pred_cls

    if target_layer is None:
        target_layer = _find_target_layer_for_gradcam(model)

    cam = GradCAM(model=model, target_layers=[target_layer])

    # Grad-CAM needs gradients
    grayscale_cam = cam(
        input_tensor=x,
        targets=[ClassifierOutputTarget(target_cls)]
    )[0]  # (H,W) in [0,1]

    rgb_float = _unnormalize_to_rgb_float(x, mean, std)
    cam_image = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)  # uint8

    return cam_image, probs, pred_cls