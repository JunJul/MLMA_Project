import argparse
import yaml
import os
import torch
import torchvision
from torch.utils.data import DataLoader
import lightning.pytorch as pl

# Import your custom data functions
from dataset import load_image, ImageDataset
from CVAE.lightning_CVAE import CheXpert_CVAEModel

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main(config_path, checkpoint_path):
    config = load_config(config_path)
    print(f"--- Starting CVAE Evaluation ---")
    print(f"Loading checkpoint: {checkpoint_path}")

    data_conf = config["data"]
    data_dir = data_conf["data_dir"]
    test_csv = os.path.join(data_dir, data_conf["test_file"])
    
    test_df = load_image(test_csv)
    test_dt = ImageDataset(test_df)
    
    # use a smaller batch size here just to grab a few images for visualization
    test_loader = DataLoader(test_dt, batch_size=8, shuffle=False, num_workers=4)

    model = CheXpert_CVAEModel.load_from_checkpoint(checkpoint_path, **config["model"]["params"])
    model.eval()

    trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)
    
    # runs the test_step() over your entire test_loader
    trainer.test(model, test_loader)

    print("\n--- Generating Visual Reconstructions ---")
    
    # one single batch from the test loader
    batch = next(iter(test_loader))
    x, meta, y = batch
    
    x = x.to(model.device)
    y = y.to(model.device)
    
    with torch.no_grad():
        # Generate the synthetic reconstructions!
        x_hat = model(x, meta, y)
    
    comparison = torch.cat([x.cpu(), x_hat.cpu()])
    
    # Save the image grid to your experiments folder
    save_dir = os.path.join(config["output_dir"], "results")
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, "reconstruction_comparison.png")
    
    torchvision.utils.save_image(comparison, image_path, nrow=8)
    print(f"Success! Saved visual comparison to: {image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Generative CVAE for CheXpert.")
    
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        default="model_confs/CVAE.yaml", 
        help="Path to the CVAE configuration YAML file."
    )
    
    parser.add_argument(
        "-ckpt", "--checkpoint", 
        type=str, 
        required=True, 
        help="Path to the trained .ckpt file (e.g., experiments/cvae/models/cvae-epoch=49-val_loss=120.5.ckpt)"
    )
    
    args = parser.parse_args()
    main(args.config, args.checkpoint)