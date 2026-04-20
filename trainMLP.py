import argparse
import os
import yaml
import json
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from tqdm import tqdm
from importlib import import_module
from pathlib import Path
import joblib
from sklearn.decomposition import PCA

from dataset import load_image, ImageDataset, CHEXPERT_CLASSES
from utils import EarlyStopping

# Base models with their config files and best checkpoints
BASE_MODELS = [
    {
        "config": "experiments/U-Zeros/exp2/ResNet50_CheXpertLoss_U-Zeros/config.yaml",
        "checkpoint": "experiments/U-Zeros/exp2/ResNet50_CheXpertLoss_U-Zeros/models/ResNet50_U-Zeros_epoch_13.pt",
    },
    {
        "config": "experiments/U-Zeros/exp2/ResNetCBAM_CheXpertLoss_U-Zeros/config.yaml",
        "checkpoint": "experiments/U-Zeros/exp2/ResNetCBAM_CheXpertLoss_U-Zeros/models/ResNetCBAM_U-Zeros_epoch_4.pt",
    },
    {
        "config": "experiments/U-Zeros/exp2/ResNetSE_CheXpertLoss_U-Zeros/config.yaml",
        "checkpoint": "experiments/U-Zeros/exp2/ResNetSE_CheXpertLoss_U-Zeros/models/ResNetSE_U-Zeros_epoch_7.pt",
    },
]


def load_trained_model(config_path, checkpoint_path, device):
    """Load a trained base model from its config and checkpoint."""
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


def generate_features(models, loader, device):
    """Run all base models on a data loader, return concatenated predictions, meta, labels."""
    all_preds, all_meta, all_labels = [], [], []

    with torch.no_grad():
        for img, meta, labels in tqdm(loader, desc="Extracting features"):
            img = img.to(device)
            batch_preds = []
            for m in models:
                logits = m(img)
                batch_preds.append(torch.sigmoid(logits).cpu())

            all_preds.append(torch.cat(batch_preds, dim=1))
            all_meta.append(meta)
            all_labels.append(labels)

    return torch.cat(all_preds), torch.cat(all_meta), torch.cat(all_labels)


def step1_generate_ensemble_data(device):
    """Generate ensemble feature CSVs from the 3 base models."""
    print("=" * 50)
    print("Step 1: Generating ensemble features")
    print("=" * 50)

    # Load all base models
    models = []
    for m_conf in BASE_MODELS:
        print(f"Loading {m_conf['checkpoint']}")
        model = load_trained_model(m_conf["config"], m_conf["checkpoint"], device)
        models.append(model)

    # Prepare data loaders (no augmentation, just eval transforms)
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_df = load_image("CheXpert-v1.0-small/train.csv")
    val_df = load_image("CheXpert-v1.0-small/valid.csv")

    train_ds = ImageDataset(train_df, eval_transform)
    val_ds = ImageDataset(val_df, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    os.makedirs("ensemble_data", exist_ok=True)

    num_preds = len(models) * 14  # 3 * 14 = 42
    pred_cols = [f"pred_{i}" for i in range(num_preds)]
    meta_cols = [f"meta_{i}" for i in range(3)]  # age, sex, view

    for split_name, loader, out_fname in [
        ("train", train_loader, "ensemble_train_features.csv"),
        ("valid", val_loader, "ensemble_valid_features.csv"),
    ]:
        print(f"\nProcessing {split_name} set...")
        preds, meta, labels = generate_features(models, loader, device)

        data = np.hstack([preds.numpy(), meta.numpy(), labels.numpy()])
        columns = pred_cols + meta_cols + list(CHEXPERT_CLASSES)
        df = pd.DataFrame(data, columns=columns)
        out_path = os.path.join("ensemble_data", out_fname)
        df.to_csv(out_path, index=False)
        print(f"Saved {out_path}: {len(df)} samples")


def find_best_pca(train_preds_np, num_preds):
    """Find smallest n_components that explains >= 95% variance."""
    candidates = [c for c in [5, 10, 15, 20, 25, 30, 35, num_preds] if c <= num_preds]

    print("\nSearching for best PCA n_components...")
    best_n = num_preds

    for n in candidates:
        pca = PCA(n_components=n)
        pca.fit(train_preds_np)
        var_explained = pca.explained_variance_ratio_.sum()
        print(f"  n_components={n:>3}, variance_explained={var_explained:.4f}")

        if var_explained >= 0.95:
            best_n = n
            break

    print(f"\nSelected n_components={best_n}")
    return best_n


def step2_train_mlp(config_path, device):
    """Train the MLP meta-learner on pre-generated ensemble features."""
    print("\n" + "=" * 50)
    print("Step 2: Training MLP Meta-Learner")
    print("=" * 50)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Data paths
    data_conf = config["data"]
    train_csv = os.path.join(data_conf["data_dir"], data_conf["train_file"])
    val_csv = os.path.join(data_conf["data_dir"], data_conf["val_file"])

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    num_models = config["model"]["params"]["num_models"]
    num_classes = config["model"]["params"]["num_classes"]
    num_meta = config["model"]["params"]["num_meta_features"]
    num_preds = num_models * num_classes  # 42

    pred_cols = [f"pred_{i}" for i in range(num_preds)]
    meta_cols = [f"meta_{i}" for i in range(num_meta)]

    train_preds_np = train_df[pred_cols].values.astype(np.float32)
    val_preds_np = val_df[pred_cols].values.astype(np.float32)
    train_meta = torch.tensor(train_df[meta_cols].values, dtype=torch.float32)
    val_meta = torch.tensor(val_df[meta_cols].values, dtype=torch.float32)
    train_labels = torch.tensor(train_df[CHEXPERT_CLASSES].values, dtype=torch.float32)
    val_labels = torch.tensor(val_df[CHEXPERT_CLASSES].values, dtype=torch.float32)

    # --- PCA on prediction features ---
    pca_conf = config.get("pca", {})
    use_pca = pca_conf.get("enabled", True)

    if use_pca:
        n_components = pca_conf.get("n_components", "auto")
        if n_components == "auto":
            n_components = find_best_pca(train_preds_np, num_preds)

        print(f"\nApplying PCA with n_components={n_components}")
        pca = PCA(n_components=n_components)
        train_preds_pca = pca.fit_transform(train_preds_np)
        val_preds_pca = pca.transform(val_preds_np)
        var_explained = pca.explained_variance_ratio_.sum()
        print(f"PCA variance explained: {var_explained:.4f}")

        train_preds = torch.tensor(train_preds_pca, dtype=torch.float32)
        val_preds = torch.tensor(val_preds_pca, dtype=torch.float32)
        pca_dim = n_components
    else:
        train_preds = torch.tensor(train_preds_np, dtype=torch.float32)
        val_preds = torch.tensor(val_preds_np, dtype=torch.float32)
        pca_dim = None

    batch_size = data_conf["batch_size"]
    train_loader = DataLoader(
        TensorDataset(train_preds, train_meta, train_labels),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_preds, val_meta, val_labels),
        batch_size=batch_size
    )

    # Model — pass pca_dim so the MLP knows the actual input size
    model_conf = config["model"]
    module = import_module(model_conf["type"])
    _, cls_name = model_conf["type"].rsplit(".", 1)
    model_params = dict(model_conf.get("params", {}))
    if pca_dim is not None:
        model_params["pca_dim"] = pca_dim
    model = getattr(module, cls_name)(**model_params)
    model = model.to(device)

    # Loss
    loss_conf = config["loss"]
    loss_module_path, loss_fn_name = loss_conf["type"].rsplit(".", 1)
    loss_module = import_module(loss_module_path)
    loss_fn = getattr(loss_module, loss_fn_name)(**loss_conf.get("params", {}))
    if isinstance(loss_fn, torch.nn.Module):
        loss_fn = loss_fn.to(device)

    # Optimizer
    opt_conf = config["optimizer"]
    optimizer = getattr(torch.optim, opt_conf["type"])(
        model.parameters(), **opt_conf["params"]
    )

    # Scheduler
    sched_conf = config["scheduler"]
    scheduler = getattr(torch.optim.lr_scheduler, sched_conf["type"])(
        optimizer, **sched_conf["params"]
    )

    # Early stopping
    es_conf = config["earlyStop"]
    es_module_path, es_name = es_conf["type"].rsplit(".", 1)
    es_module = import_module(es_module_path)
    early_stop = getattr(es_module, es_name)(**es_conf.get("params", {}))

    # Experiment directory
    exp_dir = Path(config["output_dir"])
    model_dir = exp_dir / "models"
    os.makedirs(model_dir, exist_ok=True)

    # Training loop
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    max_epochs = config.get("max_epochs", 50)
    policy = config["loss"]["params"]["policy"]

    print(f"Training MultiModalMetaLearner with params: {config['model']['params']}")
    print(f"Max epochs: {max_epochs}, Batch size: {batch_size}\n")

    for epoch in range(1, max_epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Train ---
        model.train()
        running_loss = 0.0
        for preds_b, meta_b, labels_b in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            preds_b = preds_b.to(device)
            meta_b = meta_b.to(device)
            labels_b = labels_b.to(device)

            optimizer.zero_grad()
            logits = model(preds_b, meta_b)
            loss = loss_fn(logits, labels_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # --- Validate ---
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for preds_b, meta_b, labels_b in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                preds_b = preds_b.to(device)
                meta_b = meta_b.to(device)
                labels_b = labels_b.to(device)

                logits = model(preds_b, meta_b)
                loss = loss_fn(logits, labels_b)
                running_loss += loss.item()

        val_loss = running_loss / len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch}: Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        # Scheduler step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
            }
            save_path = model_dir / f"MultiModalMetaLearner_{policy}_epoch_{epoch}.pt"
            torch.save(checkpoint, save_path)

        # Early stopping
        early_stop(val_loss)
        if early_stop.early_stop:
            print(f"Early stopping triggered at epoch {epoch}!")
            break

    # Save config and history
    config["trained_epochs"] = epoch
    config["best_epoch"] = best_epoch
    config["date"] = time.strftime("%Y%m%d")
    if use_pca:
        config["pca"] = {"enabled": True, "n_components": int(pca_dim),
                         "variance_explained": float(var_explained)}

    with open(exp_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    with open(exp_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    # Save PCA scaler for test time
    if use_pca:
        pca_path = exp_dir / "pca_scaler.pkl"
        joblib.dump(pca, pca_path)
        print(f"PCA scaler saved to {pca_path}")

    print(f"\nTraining complete. Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")
    print(f"Results saved to {exp_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train MLP ensemble meta-learner")
    parser.add_argument(
        "-c", "--config", type=str, default="model_confs/MLP.yaml",
        help="Path to MLP configuration YAML file."
    )
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="Skip feature generation step (use if ensemble_data already exists)."
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not args.skip_generate:
        step1_generate_ensemble_data(device)

    step2_train_mlp(args.config, device)


if __name__ == "__main__":
    main()
