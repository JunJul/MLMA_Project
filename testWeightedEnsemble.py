"""
Weighted Ensemble: combine 3 base models using per-class AUROC weights.
Each model's prediction for each class is weighted by its validation AUROC on that class.
No training required — weights are derived directly from validation performance.
"""
import argparse
import os
import yaml

import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from importlib import import_module
from pathlib import Path
from sklearn.metrics import roc_auc_score

from dataset import load_image, ImageDataset, CHEXPERT_CLASSES
from metrics import classification_result

# Base models: config + checkpoint
BASE_MODELS = [
    {
        "name": "ResNet50",
        "config": "experiments/U-Zeros/exp2/ResNet50_CheXpertLoss_U-Zeros/config.yaml",
        "checkpoint": "experiments/U-Zeros/exp2/ResNet50_CheXpertLoss_U-Zeros/models/ResNet50_U-Zeros_epoch_13.pt",
    },
    {
        "name": "ResNetCBAM",
        "config": "experiments/U-Zeros/exp2/ResNetCBAM_CheXpertLoss_U-Zeros/config.yaml",
        "checkpoint": "experiments/U-Zeros/exp2/ResNetCBAM_CheXpertLoss_U-Zeros/models/ResNetCBAM_U-Zeros_epoch_4.pt",
    },
    {
        "name": "ResNetSE",
        "config": "experiments/U-Zeros/exp2/ResNetSE_CheXpertLoss_U-Zeros/config.yaml",
        "checkpoint": "experiments/U-Zeros/exp2/ResNetSE_CheXpertLoss_U-Zeros/models/ResNetSE_U-Zeros_epoch_7.pt",
    },
]


def load_trained_model(config_path, checkpoint_path, device):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_conf = config["model"]
    module_path = model_conf["type"]
    _, cls_name = module_path.rsplit(".", 1)
    module = import_module(module_path)
    model = getattr(module, cls_name)(**model_conf.get("params", {}))
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def get_predictions(model, loader, device):
    """Get sigmoid probabilities and labels from a model."""
    all_probs, all_labels = [], []
    with torch.no_grad():
        for img, _, labels in tqdm(loader):
            img = img.to(device)
            logits = model(img)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_per_class_auroc(y_true, y_probs):
    """Compute AUROC for each of the 14 classes."""
    aurocs = np.zeros(len(CHEXPERT_CLASSES))
    for i, cls in enumerate(CHEXPERT_CLASSES):
        try:
            aurocs[i] = roc_auc_score(y_true[:, i], y_probs[:, i])
        except ValueError:
            aurocs[i] = 0.5  # default for classes with no positive samples
    return aurocs


def main():
    parser = argparse.ArgumentParser(description="AUROC-weighted ensemble of 3 models")
    parser.add_argument("--val-csv", type=str, default="CheXpert-v1.0-small/valid.csv",
                        help="Validation CSV to compute weights from")
    parser.add_argument("--test-csv", type=str, default="CheXpert-v1.0-small/valid.csv",
                        help="Test CSV to evaluate on")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="experiments/U-Zeros/weighted_ensemble")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data loaders
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_df = load_image(args.val_csv)
    test_df = load_image(args.test_csv)

    val_loader = DataLoader(ImageDataset(val_df, eval_transform), batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(ImageDataset(test_df, eval_transform), batch_size=args.batch_size, num_workers=0)

    # Load models
    models = []
    for m_conf in BASE_MODELS:
        print(f"Loading {m_conf['name']}: {m_conf['checkpoint']}")
        model = load_trained_model(m_conf["config"], m_conf["checkpoint"], device)
        models.append(model)

    # Step 1: Get each model's predictions on validation set to compute weights
    print("\n=== Computing per-class AUROC weights from validation set ===")
    val_aurocs = []  # shape: [num_models, 14]
    y_true_val = None

    for i, (model, m_conf) in enumerate(zip(models, BASE_MODELS)):
        print(f"\n{m_conf['name']} predictions on val set:")
        probs, labels = get_predictions(model, val_loader, device)
        if y_true_val is None:
            y_true_val = labels
        aurocs = compute_per_class_auroc(y_true_val, probs)
        val_aurocs.append(aurocs)

        for j, cls in enumerate(CHEXPERT_CLASSES):
            print(f"  {cls:>30}: {aurocs[j]:.4f}")

    val_aurocs = np.array(val_aurocs)  # [3, 14]

    # Normalize weights per class (so they sum to 1 for each class)
    weights = val_aurocs / val_aurocs.sum(axis=0, keepdims=True)  # [3, 14]

    print("\n=== Normalized per-class weights ===")
    header = f"{'Class':>30} | {'ResNet50':>10} | {'CBAM':>10} | {'SE':>10}"
    print(header)
    print("-" * len(header))
    for j, cls in enumerate(CHEXPERT_CLASSES):
        print(f"{cls:>30} | {weights[0, j]:>10.4f} | {weights[1, j]:>10.4f} | {weights[2, j]:>10.4f}")

    # Step 2: Get predictions on test set and apply weighted average
    print("\n=== Generating weighted ensemble predictions on test set ===")
    test_preds_list = []
    y_true_test = None

    for i, (model, m_conf) in enumerate(zip(models, BASE_MODELS)):
        print(f"{m_conf['name']} predictions on test set...")
        probs, labels = get_predictions(model, test_loader, device)
        if y_true_test is None:
            y_true_test = labels
        test_preds_list.append(probs)

    test_preds = np.stack(test_preds_list)  # [3, N, 14]

    # Weighted combination: for each class, weight each model's prediction by its AUROC weight
    # weights shape: [3, 14] -> [3, 1, 14] for broadcasting with [3, N, 14]
    weighted_preds = (test_preds * weights[:, np.newaxis, :]).sum(axis=0)  # [N, 14]

    # Save results
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Save weights for reference
    weights_dict = {}
    for j, cls in enumerate(CHEXPERT_CLASSES):
        weights_dict[cls] = {
            BASE_MODELS[i]["name"]: float(weights[i, j]) for i in range(len(models))
        }
    with open(output_dir / "ensemble_weights.yaml", "w") as f:
        yaml.dump(weights_dict, f, default_flow_style=False)

    print(f"\nWeights saved to {output_dir / 'ensemble_weights.yaml'}")

    # Classification report
    model_name = "Weighted_Ensemble"
    classification_result(y_true_test, weighted_preds, model_name, output_dir)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
