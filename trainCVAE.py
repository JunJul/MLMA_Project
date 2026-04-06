import argparse
import yaml
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# Import your custom data functions
from dataset import load_image, ImageDataset
from CVAE.lightning_CVAE import CVAEModel


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(args):
    config_path = args.config
    config = load_config(config_path)
    print(f"--- Starting CVAE Training using {config_path} ---")

    # Optional: faster matrix multiplication on supported GPUs
    torch.set_float32_matmul_precision('medium')

    # -------------------- Data Setup --------------------
    data_conf = config["data"]
    data_dir = data_conf["data_dir"]

    train_csv = os.path.join(data_dir, data_conf["train_file"])
    val_csv = os.path.join(data_dir, data_conf["val_file"])

    train_df = load_image(train_csv)

    if data_conf["val_file"] != "None" and os.path.exists(val_csv):
        val_df = load_image(val_csv)
    else:
        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # Transforms: augmentation only for training
    cvae_train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    cvae_val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dt = ImageDataset(train_df, transform=cvae_train_transform)
    val_dt = ImageDataset(val_df, transform=cvae_val_transform)

    train_loader = DataLoader(
        train_dt,
        batch_size=data_conf["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dt,
        batch_size=data_conf["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # -------------------- Model Setup --------------------
    model_params = config["model"]["params"]
    if args.policy is not None:
        model_params["policy"] = args.policy

    policy_name = model_params.get("policy", "unknown")
    print(f"Training with Policy: {policy_name}")

    model = CVAEModel(**model_params)

    # -------------------- Callbacks --------------------
    early_stop = EarlyStopping(
        monitor="val_ssim_score",
        patience=config["earlyStop"]["params"]["patience"],
        mode="max",
        verbose=True
    )

    # Create output directory if it doesn't exist
    output_root = config["output_dir"]
    os.makedirs(output_root, exist_ok=True)
    save_dir = os.path.join(output_root, f"CVAE_{policy_name}", "models")
    os.makedirs(save_dir, exist_ok=True)

    filename = f"cvae-{policy_name}"
    checkpoint = ModelCheckpoint(
        filename=filename + "-{epoch:02d}-{val_ssim_score:.4f}",
        save_top_k=1,
        monitor="val_ssim_score",
        mode="max"
    )

    # -------------------- Trainer --------------------
    # Use the same root directory as save_dir's parent for logs
    log_root = os.path.join(output_root, f"CVAE_{policy_name}")
    trainer = pl.Trainer(
        max_epochs=config.get("max_epochs", 50),
        accelerator="auto",
        devices=1,
        accumulate_grad_batches=2,
        gradient_clip_val=1.0,          # Built‑in gradient clipping
        callbacks=[early_stop, checkpoint],
        default_root_dir=log_root,
        precision='16-mixed',
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)

    print("--- CVAE Training Complete! ---")
    print(f"Best model saved at: {checkpoint.best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Generative CVAE for CheXpert.")
    parser.add_argument("-c", "--config", type=str, default="model_confs/CVAE.yaml",
                        help="Path to YAML config file")
    parser.add_argument(
        "-p", "--policy",
        type=str,
        choices=["U-Ones", "U-Zeros", "U-Smooth"],
        default=None,
        help="Override the CheXpert uncertain label policy."
    )
    args = parser.parse_args()
    main(args)