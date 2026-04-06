import argparse
import yaml
import os
import glob
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from dataset import load_image, ImageDataset
from CVAE.lightning_CVAE import CVAEModel


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(args):
    config = load_config(args.config)
    policy_name = args.policy
    print(f"--- Starting CVAE Evaluation for Policy: {policy_name} ---")

    # ==========================================
    # 1. AUTOMATICALLY FIND THE LATEST CHECKPOINT (Lightning Structure)
    # ==========================================
    # We look for the base directory for this specific policy
    # Note: Adjust 'config["output_dir"]' if it doesn't point to the parent of CVAE_U-Ones
    policy_dir = os.path.join(config.get("output_dir", "experiments/cvae"), f"CVAE_{policy_name}")
    
    # Recursively search for all .ckpt files inside lightning_logs, version_X, checkpoints, etc.
    search_pattern = os.path.join(policy_dir, "**", "*.ckpt")
    ckpt_files = glob.glob(search_pattern, recursive=True)
    
    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"Could not find any .ckpt files anywhere under {policy_dir}!")

    # Standard PyTorch Lightning behavior: prefer 'last.ckpt' if you have save_last=True
    last_ckpts = [f for f in ckpt_files if f.endswith("last.ckpt")]
    
    if last_ckpts:
        # If there are multiple last.ckpt (from different versions), get the most recently modified one
        last_ckpts.sort(key=os.path.getmtime, reverse=True)
        checkpoint_path = last_ckpts[0]
    else:
        # Fallback: Sort all found .ckpt files by modification time (newest first)
        ckpt_files.sort(key=os.path.getmtime, reverse=True)
        checkpoint_path = ckpt_files[0]
        
    print(f"Automatically found latest checkpoint: {checkpoint_path}")

    # ==========================================
    # 2. SETUP DATA
    # ==========================================
    data_conf = config["data"]
    data_dir = data_conf["data_dir"]
    test_csv = os.path.join(data_dir, data_conf["test_file"])
    
    cvae_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    test_df = load_image(test_csv)
    test_dt = ImageDataset(test_df, transform=cvae_test_transform)
    test_loader = DataLoader(test_dt, batch_size=8, shuffle=False, num_workers=4)

    # ==========================================
    # 3. LOAD MODEL (directly to device)
    # ==========================================
    model_params = config["model"]["params"]
    model_params["policy"] = policy_name

    # Determine device first
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint directly to the target device
    model = CVAEModel.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        **model_params
    )
    model = model.to(device)
    model.eval()

    # ==========================================
    # 4. RUN TEST SET
    # ==========================================
    trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)
    trainer.test(model, test_loader)
    # Trainer may move the model during testing and then restore it to CPU.
    # Ensure the model is back on the target device for manual inference.
    model = model.to(device)
    model.eval()

    # ==========================================
    # 5. GENERATE VISUAL COMPARISON
    # ==========================================
    print("\n--- Generating Visual Reconstructions ---")
    
    batch = next(iter(test_loader))
    x, meta, y = batch
    x = x.to(device, dtype=torch.float32)
    meta = meta.to(device, dtype=torch.float32)
    y = y.to(device, dtype=torch.float32)
    
    with torch.no_grad():
        x_hat, _ = model(x, meta, y)   # Unpack (reconstruction, kl)
    
    # Clamp to valid range for saving (model output already has sigmoid, but safety)
    x_hat = torch.clamp(x_hat, 0.0, 1.0)
    
    # Concatenate original and reconstructed
    comparison = torch.cat([x.cpu(), x_hat.cpu()])
    
    save_dir = os.path.join(policy_dir, "results")
    os.makedirs(save_dir, exist_ok=True)
    
    image_path = os.path.join(save_dir, "reconstruction_comparison.png")
    torchvision.utils.save_image(comparison, image_path, nrow=8)
    
    print(f"Saved visual comparison to: {image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Generative CVAE for CheXpert.")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="model_confs/CVAE.yaml",
        help="Path to the CVAE configuration YAML file."
    )
    parser.add_argument(
        "-p", "--policy",
        type=str,
        choices=["U-Ones", "U-Zeros", "U-Smooth"],
        default="U-Ones",
        help="Which policy's checkpoint to load."
    )
    args = parser.parse_args()
    main(args)