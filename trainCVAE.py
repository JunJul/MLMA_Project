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

    # Setup data
    data_conf = config["data"]
    data_dir = data_conf["data_dir"]
    
    # Load DataFrames
    train_csv = os.path.join(data_dir, data_conf["train_file"])
    val_csv = os.path.join(data_dir, data_conf["val_file"])
    
    train_df = load_image(train_csv)
    
    # Handle validation split
    if data_conf["val_file"] != "None" and os.path.exists(val_csv):
        val_df = load_image(val_csv)
    else:
        train_df, val_df = train_test_split(train_df, test_size=0.2)

    cvae_train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    
    # Validation data doesn't need data augmentation (flipping/rotating)
    cvae_val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Create PyTorch DataLoaders and pass the custom transforms!
    train_dt = ImageDataset(train_df, transform=cvae_train_transform)
    val_dt = ImageDataset(val_df, transform=cvae_val_transform)
    
    train_loader = DataLoader(train_dt, batch_size=data_conf["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dt, batch_size=data_conf["batch_size"], shuffle=False, num_workers=4)

    
    # Initialize CVAE
    model_params = config["model"]["params"]
    if args.policy is not None:
        model_params["policy"] = args.policy
        
    policy_name = model_params.get("policy", "unknown")
    print(f"Training with Policy: {policy_name}")
    
    model = CVAEModel(**model_params)

    # Call Back
    early_stop = EarlyStopping(
        monitor="val_loss", 
        patience=config["earlyStop"]["params"]["patience"],
        mode="min",
        verbose=True
    )
    
    save_dir = os.path.join(config["output_dir"], f"CVAE_{policy_name}", "models")
    
    filename = f"cvae-{policy_name}"
    checkpoint = ModelCheckpoint(
        dirpath=save_dir,
        filename=filename + "-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    # Train CVAE
    out_dir = config["output_dir"] + f"_{policy_name}"
    trainer = pl.Trainer(
        max_epochs=config.get("max_epochs", 50),
        accelerator="auto",
        devices=1,
        callbacks=[early_stop, checkpoint],
        default_root_dir=out_dir
    )


    trainer.fit(model, train_loader, val_loader)
    
    print("--- CVAE Training Complete! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Generative CVAE for CheXpert.")
    parser.add_argument("-c", "--config", type=str, default="model_confs/CVAE.yaml")
    
    parser.add_argument(
        "-p", "--policy",
        type=str,
        choices=["U-Ones", "U-Zeros", "U-Smooth"], 
        default=None, 
        help="Override the CheXpert loss policy."
    )
    
    args = parser.parse_args()
    main(args)