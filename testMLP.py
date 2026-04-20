import argparse
import os
import yaml

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from importlib import import_module
from pathlib import Path
import joblib

from dataset import CHEXPERT_CLASSES
from metrics import classification_result


def main(config_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Use experiment config if it exists (has best_epoch, pca info, etc.)
    exp_dir = Path(config["output_dir"])
    exp_config_path = exp_dir / "config.yaml"
    if exp_config_path.exists():
        print(f"Loading experiment config from {exp_config_path}")
        with open(exp_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    # Data
    data_conf = config["data"]
    test_csv = os.path.join(data_conf["data_dir"], data_conf["test_file"])
    test_df = pd.read_csv(test_csv)

    num_models = config["model"]["params"]["num_models"]
    num_classes = config["model"]["params"]["num_classes"]
    num_meta = config["model"]["params"]["num_meta_features"]
    num_preds = num_models * num_classes

    pred_cols = [f"pred_{i}" for i in range(num_preds)]
    meta_cols = [f"meta_{i}" for i in range(num_meta)]

    test_preds_np = test_df[pred_cols].values.astype(np.float32)
    test_meta = torch.tensor(test_df[meta_cols].values, dtype=torch.float32)
    test_labels = torch.tensor(test_df[CHEXPERT_CLASSES].values, dtype=torch.float32)

    # PCA transform if enabled
    exp_dir = Path(config["output_dir"])
    pca_conf = config.get("pca", {})
    pca_dim = None
    if pca_conf.get("enabled", False):
        pca_path = exp_dir / "pca_scaler.pkl"
        print(f"Loading PCA scaler from {pca_path}")
        pca = joblib.load(pca_path)
        test_preds_np = pca.transform(test_preds_np)
        pca_dim = pca_conf["n_components"]
        print(f"PCA applied: {num_preds} -> {pca_dim} features")

    test_preds = torch.tensor(test_preds_np, dtype=torch.float32)

    batch_size = data_conf["batch_size"]
    test_loader = DataLoader(
        TensorDataset(test_preds, test_meta, test_labels),
        batch_size=batch_size
    )

    # Model
    model_conf = config["model"]
    module = import_module(model_conf["type"])
    _, cls_name = model_conf["type"].rsplit(".", 1)
    model_params = dict(model_conf.get("params", {}))
    if pca_dim is not None:
        model_params["pca_dim"] = pca_dim
    model = getattr(module, cls_name)(**model_params)
    model = model.to(device)

    # Load best checkpoint
    exp_dir = Path(config["output_dir"])
    best_epoch = config.get("best_epoch", 0)
    policy = config["loss"]["params"]["policy"]
    ckpt_path = exp_dir / "models" / f"MultiModalMetaLearner_{policy}_epoch_{best_epoch}.pt"

    if not ckpt_path.exists():
        # Try to find any checkpoint in the models dir
        model_dir = exp_dir / "models"
        pts = sorted(model_dir.glob("*.pt"))
        if pts:
            ckpt_path = pts[-1]
            print(f"best_epoch checkpoint not found, using: {ckpt_path}")
        else:
            raise FileNotFoundError(f"No checkpoints found in {model_dir}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Loss
    loss_conf = config["loss"]
    loss_module_path, loss_fn_name = loss_conf["type"].rsplit(".", 1)
    loss_module = import_module(loss_module_path)
    loss_fn = getattr(loss_module, loss_fn_name)(**loss_conf.get("params", {}))
    if isinstance(loss_fn, torch.nn.Module):
        loss_fn = loss_fn.to(device)

    # Predict
    print("------ Start Testing ------")
    model.eval()
    running_loss = 0.0
    all_probs, all_targets = [], []

    with torch.no_grad():
        for preds_b, meta_b, labels_b in tqdm(test_loader, desc="Testing"):
            preds_b = preds_b.to(device)
            meta_b = meta_b.to(device)
            labels_b = labels_b.to(device)

            logits = model(preds_b, meta_b)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_targets.append(labels_b.cpu())

            loss = loss_fn(logits, labels_b)
            running_loss += loss.item()

    avg_loss = running_loss / len(test_loader)
    y_probs = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_targets).numpy()

    print(f"\nTesting Completed. Average Test Loss: {avg_loss:.4f}")

    model_name = "MLP_Ensemble"
    classification_result(y_true, y_probs, model_name, exp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the MLP ensemble meta-learner.")
    parser.add_argument(
        "-c", "--config", type=str, default="model_confs/MLP.yaml",
        help="Path to MLP configuration YAML file."
    )
    args = parser.parse_args()
    main(args.config)
