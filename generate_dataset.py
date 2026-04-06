"""
Generate synthetic chest X-ray images using CVAE to balance the CheXpert training dataset.

This script:
1. Analyzes class distribution in the training dataset
2. Loads a trained CVAE model
3. Generates synthetic images for underrepresented diseases
4. Saves generated images and updates the training CSV
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, List, Dict
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# Import project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from CVAE.lightning_CVAE import CVAEModel
from dataset import CHEXPERT_CLASSES


def analyze_class_distribution(csv_path: str) -> Dict[str, float]:
    """
    Analyze the class distribution in the CheXpert dataset.
    
    Args:
        csv_path: Path to the training CSV file
        
    Returns:
        Dictionary with disease names as keys and positive sample ratios as values
    """
    df = pd.read_csv(csv_path)
    
    # Fill NaN with 0.0 (negative)
    df[CHEXPERT_CLASSES] = df[CHEXPERT_CLASSES].fillna(0.0)
    
    # Calculate positive ratio for each disease
    class_distribution = {}
    total_samples = len(df)
    
    for disease in CHEXPERT_CLASSES:
        positive_samples = (df[disease] == 1.0).sum()
        ratio = positive_samples / total_samples
        class_distribution[disease] = ratio
        
    return class_distribution, total_samples


def print_class_distribution(distribution: Dict[str, float], total: int):
    """Print class distribution in a readable format."""
    print("\n" + "="*70)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*70)
    
    sorted_dist = sorted(distribution.items(), key=lambda x: x[1])
    
    for disease, ratio in sorted_dist:
        count = int(ratio * total)
        bar = "█" * int(ratio * 50)
        print(f"{disease:35} | {count:5d} ({ratio*100:5.2f}%) | {bar}")
    
    print("="*70 + "\n")


def get_underrepresented_classes(
    distribution: Dict[str, float],
    percentile_threshold: float = 50
) -> List[str]:
    """
    Identify underrepresented disease classes.
    
    Args:
        distribution: Class distribution dictionary
        percentile_threshold: Classes below this percentile are considered underrepresented
        
    Returns:
        List of underrepresented disease names
    """
    ratios = list(distribution.values())
    threshold = np.percentile(ratios, percentile_threshold)
    
    underrepresented = [
        disease for disease, ratio in distribution.items()
        if ratio < threshold
    ]
    
    return underrepresented


def create_label_tensor(disease_labels: List[float]) -> torch.Tensor:
    """Convert disease labels list to tensor."""
    return torch.tensor(disease_labels, dtype=torch.float32)


def create_meta_tensor(age: float, sex: str, view: str) -> torch.Tensor:
    """
    Create metadata tensor from patient information.
    
    Args:
        age: Patient age (will be normalized by 100)
        sex: 'Male' or 'Female'
        view: 'Frontal', 'Lateral', 'AP', or 'PA'
        
    Returns:
        Tensor with shape (3,) containing [age_norm, sex_code, view_code]
    """
    # Age normalization
    age_normalized = min(age / 100.0, 1.0) if age > 0 else 0.5
    
    # Sex encoding
    sex_code = 1.0 if sex.lower() == 'female' else 0.0
    
    # View encoding: 0=Lateral, 0.5=PA, 1.0=AP
    if view.lower() == 'lateral':
        view_code = 0.0
    elif view.lower() == 'pa':
        view_code = 0.5
    elif view.lower() == 'ap':
        view_code = 1.0
    else:
        view_code = 0.5  # default
    
    return torch.tensor([age_normalized, sex_code, view_code], dtype=torch.float32)


def generate_synthetic_images(
    cvae_model: CVAEModel,
    disease_labels: torch.Tensor,
    metadata: torch.Tensor,
    num_samples: int = 1,
    device: str = 'cuda',
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Generate synthetic chest X-ray images from the CVAE.
    
    Args:
        cvae_model: Trained CVAE model
        disease_labels: Target disease labels (1D tensor of shape (14,))
        metadata: Patient metadata (1D tensor of shape (3,))
        num_samples: Number of images to generate
        device: Device to use ('cuda' or 'cpu')
        temperature: Temperature for sampling (higher = more variation)
        
    Returns:
        Generated images tensor of shape (num_samples, 3, 224, 224)
    """
    cvae_model.eval()
    cvae_model = cvae_model.to(device)
    
    with torch.no_grad():
        # Prepare inputs
        y_processed = cvae_model.apply_policy(disease_labels.unsqueeze(0))
        y_embed = cvae_model.get_multi_label_embedding(y_processed)
        
        meta_expanded = metadata.unsqueeze(0).to(device)
        meta_embed = torch.relu(cvae_model.embed_meta(meta_expanded.float()))
        
        # Sample from latent space
        latent_dims = cvae_model.hparams.latent_dims
        z = torch.randn(num_samples, latent_dims, device=device) * temperature
        
        # Expand embeddings to match batch size
        y_embed_expanded = y_embed.expand(num_samples, -1)
        meta_embed_expanded = meta_embed.expand(num_samples, -1)
        
        # Decode to images
        generated_images = cvae_model.cvae.decoder(z)
    
    return generated_images.cpu()


def save_generated_images(
    images: torch.Tensor,
    output_dir: str,
    disease_name: str,
    disease_idx: int,
    start_idx: int = 0,
    base_patient_id: str = "synthetic"
) -> List[Tuple[str, str]]:
    """
    Save generated images to disk.
    
    Args:
        images: Tensor of images with shape (num_samples, 3, 224, 224)
        output_dir: Directory to save images
        disease_name: Name of the disease being generated for
        disease_idx: Index of the disease in CHEXPERT_CLASSES
        start_idx: Starting index for image numbering
        base_patient_id: Base identifier for synthetic patients
        
    Returns:
        List of tuples (relative_path, image_desc) for CSV entries
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create disease-specific subdirectory
    disease_dir = output_dir / disease_name.replace(' ', '_')
    disease_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for i, img_tensor in enumerate(images):
        # Convert tensor to PIL Image
        # Tensor shape: (3, 224, 224), values in [0, 1]
        img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_array)
        
        # Create filename
        filename = f"{base_patient_id}_{disease_idx:02d}_{start_idx + i:05d}.jpg"
        filepath = disease_dir / filename
        
        # Save image
        img_pil.save(filepath)
        
        # Store relative path for CSV
        relative_path = str(filepath.relative_to(output_dir.parent))
        saved_paths.append((relative_path, disease_name, start_idx + i))
    
    return saved_paths


def create_synthetic_csv_entries(
    saved_paths: List[Tuple[str, str, int]],
    disease_idx: int,
    age: float = 50.0,
    sex: str = 'Female',
    view: str = 'Frontal'
) -> List[Dict]:
    """
    Create CSV entries for generated images.
    
    Args:
        saved_paths: List of (path, disease_name, idx) tuples
        disease_idx: Index of the disease in CHEXPERT_CLASSES
        age: Patient age
        sex: Patient sex
        view: Image view
        
    Returns:
        List of dictionaries with CSV row data
    """
    csv_entries = []
    
    for path, disease_name, idx in saved_paths:
        # Initialize row with all diseases as NaN (not mentioned)
        row = {'Path': path, 'Sex': sex, 'Age': age, 'Frontal/Lateral': view}
        
        # Determine AP/PA
        if view.lower() == 'lateral':
            row['AP/PA'] = None
        else:
            # Randomly choose AP or PA for Frontal view
            row['AP/PA'] = np.random.choice(['AP', 'PA'])
        
        # Set disease label to 1.0 (positive), others to NaN
        for disease in CHEXPERT_CLASSES:
            if disease == disease_name:
                row[disease] = 1.0
            else:
                row[disease] = np.nan
        
        csv_entries.append(row)
    
    return csv_entries


def balance_dataset(
    csv_path: str,
    checkpoint_path: str,
    output_dir: str = "generated_images",
    target_ratio: float = 0.15,
    device: str = 'cuda',
    num_samples_per_class: int = None,
    temperature: float = 1.0,
    save_new_csv: bool = True
):
    """
    Main function to balance the CheXpert dataset with CVAE-generated images.
    
    Args:
        csv_path: Path to training CSV
        checkpoint_path: Path to trained CVAE checkpoint
        output_dir: Directory to save generated images
        target_ratio: Target positive ratio for underrepresented classes
        device: Device to use ('cuda' or 'cpu')
        num_samples_per_class: Override automatic calculation with fixed number
        temperature: Temperature for latent sampling
        save_new_csv: Whether to save updated CSV with synthetic samples
    """
    print("\n" + "="*70)
    print("CVAE-BASED DATASET BALANCING")
    print("="*70)
    
    # Step 1: Analyze current distribution
    print("\n[1/5] Analyzing current class distribution...")
    distribution, total_samples = analyze_class_distribution(csv_path)
    print_class_distribution(distribution, total_samples)
    
    # Step 2: Identify underrepresented classes
    print("[2/5] Identifying underrepresented classes...")
    underrepresented = get_underrepresented_classes(distribution, percentile_threshold=50)
    print(f"Found {len(underrepresented)} underrepresented classes:")
    for disease in underrepresented:
        print(f"  - {disease}: {distribution[disease]*100:.2f}%")
    
    # Step 3: Load CVAE model
    print(f"\n[3/5] Loading CVAE model from {checkpoint_path}...")
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU instead")
        device = 'cpu'
    
    cvae_model = CVAEModel.load_from_checkpoint(checkpoint_path)
    cvae_model = cvae_model.to(device)
    print(f"Model loaded successfully on {device.upper()}")
    
    # Step 4: Generate synthetic images
    print(f"\n[4/5] Generating synthetic images...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_csv_entries = []
    
    for disease_idx, disease in enumerate(underrepresented):
        # Calculate number of samples needed
        if num_samples_per_class:
            n_generate = num_samples_per_class
        else:
            current_count = int(distribution[disease] * total_samples)
            target_count = int(target_ratio * total_samples)
            n_generate = max(0, target_count - current_count)
        
        if n_generate <= 0:
            print(f"  Skipping {disease} (already well-represented)")
            continue
        
        print(f"\n  Generating {n_generate} images for '{disease}'...")
        
        # Create disease labels (one-hot encoding)
        disease_labels = torch.zeros(14, dtype=torch.float32)
        disease_labels[CHEXPERT_CLASSES.index(disease)] = 1.0
        
        # Random metadata for diversity
        metadata_list = []
        for _ in range(n_generate):
            age = np.random.uniform(40, 80)
            sex = np.random.choice(['Male', 'Female'])
            view = np.random.choice(['Frontal'], p=[1.0])  # Mostly frontal
            meta = create_meta_tensor(age, sex, view)
            metadata_list.append((meta, age, sex, view))
        
        # Generate images in batches
        batch_size = 8
        saved_paths = []
        
        for batch_idx in range(0, n_generate, batch_size):
            batch_end = min(batch_idx + batch_size, n_generate)
            batch_size_actual = batch_end - batch_idx
            
            # Generate images
            images = generate_synthetic_images(
                cvae_model,
                disease_labels,
                metadata_list[batch_idx][0],
                num_samples=batch_size_actual,
                device=device,
                temperature=temperature
            )
            
            # Save images
            paths = save_generated_images(
                images,
                output_dir,
                disease,
                disease_idx,
                start_idx=batch_idx,
                base_patient_id="synthetic"
            )
            saved_paths.extend(paths)
            
            print(f"    Saved {len(paths)} images ({batch_idx}/{n_generate})")
        
        # Create CSV entries
        entries = create_synthetic_csv_entries(
            saved_paths,
            disease_idx,
            age=metadata_list[0][1],
            sex=metadata_list[0][2],
            view=metadata_list[0][3]
        )
        all_csv_entries.extend(entries)
        
        print(f"  ✓ Generated {len(saved_paths)} images for '{disease}'")
    
    # Step 5: Update CSV and save
    print(f"\n[5/5] Updating training CSV...")
    if save_new_csv and all_csv_entries:
        df_original = pd.read_csv(csv_path)
        df_synthetic = pd.DataFrame(all_csv_entries)
        
        # Combine dataframes
        df_balanced = pd.concat([df_original, df_synthetic], ignore_index=True)
        
        # Save new CSV
        output_csv = Path(csv_path).parent / "train_balanced.csv"
        df_balanced.to_csv(output_csv, index=False)
        
        print(f"✓ Saved balanced CSV to {output_csv}")
        print(f"  Original samples: {len(df_original)}")
        print(f"  Synthetic samples: {len(df_synthetic)}")
        print(f"  Total samples: {len(df_balanced)}")
    
    print("\n" + "="*70)
    print("BALANCING COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic chest X-ray images to balance CheXpert dataset"
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        default="CheXpert-v1.0-small/train.csv",
        help="Path to training CSV file"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/cvae/checkpoints/last.ckpt",
        help="Path to trained CVAE checkpoint"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_images",
        help="Directory to save generated images"
    )
    
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.15,
        help="Target positive ratio for underrepresented classes (0-1)"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Fixed number of samples to generate per class (overrides target-ratio)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for latent sampling (higher = more variation)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for generation"
    )
    
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze distribution without generating images"
    )
    
    args = parser.parse_args()
    
    # Check if CSV exists
    if not os.path.exists(args.csv):
        print(f"ERROR: CSV file not found: {args.csv}")
        sys.exit(1)
    
    # If analyze-only, just print distribution
    if args.analyze_only:
        distribution, total = analyze_class_distribution(args.csv)
        print_class_distribution(distribution, total)
        return
    
    # Check if checkpoint exists and provide helpful suggestions
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        # Try to find checkpoint files automatically
        print(f"\nERROR: Checkpoint file not found at: {checkpoint_path}\n")
        
        # Search for available checkpoints
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = []
        
        for root, dirs, files in os.walk(os.path.join(base_dir, "experiments")):
            for file in files:
                if file.endswith(".ckpt"):
                    full_path = os.path.join(root, file)
                    candidates.append(full_path)
        
        if candidates:
            print("Available checkpoint files found:")
            for i, path in enumerate(candidates, 1):
                # Show relative path for readability
                rel_path = os.path.relpath(path, base_dir)
                print(f"  {i}. {rel_path}")
            
            print(f"\nUsage example:")
            print(f"  python generate_dataset.py --checkpoint \"{candidates[0]}\"")
        else:
            print("No checkpoint files found in experiments/ directory.")
            print("Please train a CVAE model first using trainCVAE.py")
        
        sys.exit(1)
    
    # Run balancing
    balance_dataset(
        csv_path=args.csv,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        target_ratio=args.target_ratio,
        device=args.device,
        num_samples_per_class=args.num_samples,
        temperature=args.temperature,
        save_new_csv=True
    )


if __name__ == "__main__":
    main()
