# CheXpert Challenge

Multi-label chest X-ray disease classification on the [CheXpert](https://www.kaggle.com/datasets/ashery/chexpert) dataset with CVAE-based data augmentation, attention-enhanced ResNet models, ensemble uncertainty quantification, and Grad-CAM interpretability.

> **Note:** This project uses the small subset from Kaggle. The full dataset is available at [Stanford AIMI](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays).

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Data Augmentation (CVAE)](#data-augmentation-cvae)
- [Classification Models](#classification-models)
- [PCA + Ensemble](#pca--ensemble)
- [Weighted Ensemble](#weighted-ensemble)
- [Training & Testing](#training--testing)
- [Pretrained Models](#pretrained-models)

---

## Overview

**Goal:** Improve chest X-ray disease classification on imbalanced CheXpert data by:

1. **Addressing class imbalance** via Conditional VAE (CVAE)-generated synthetic images
2. **Comparing CNN with attention Models** (CBAM vs SE) against a standard ResNet-50 baseline
3. **Ensemble + weighted Ensemble Model** combine performance of three different CNNs

**Workflow:**
```
Data Analysis → CVAE Training → Synthetic Image Generation → Model Training
→ Ensemble Training → Evaluation & Metrics → Uncertainty Detection → UI Deployment
```

**Target Classes (14):**
```python
CHEXPERT_CLASSES = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]
```

**Competition Tasks:** Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion

---

## Project Structure

```
├── analyze_distribution.py    # Analyze class distribution & imbalance in training data
├── dataset.py                 # ImageDataset class (loads CheXpert CSV, handles labels/metadata)
├── generate_dataset.py        # Generate synthetic X-rays from a trained CVAE checkpoint
├── losses.py                  # CheXpertLoss with uncertain label policies (U-Ones/Zeros/Smooth/Ignore)
├── metrics.py                 # Classification reports, ROC curves, AUROC scores
├── pipeline.py                # Central training/evaluation orchestration (loaders, training loops, checkpointing)
├── trainCVAE.py               # Train CVAE with PyTorch Lightning
├── trainModels.py             # Train classification models (ResNet variants)
├── trainMLP.py                # Train ensemble gating MLP
├── trainWeightedEnsemble      # Train weighted ensemble model
├── testWeightedEnsemble       # Evaluate weighted ensemble model
├── testCVAE.py                # Evaluate CVAE reconstruction & generation quality
├── testModels.py              # Evaluate classification models (metrics & visualization)
├── user_interface.py          # Streamlit web app for predictions & interpretability
├── utils.py                   # Utilities (EarlyStopping, label smoothing loss, Grad-CAM overlay)
│
├── CVAE/
│   ├── CVAE.py                # Conditional VAE architecture (encoder + decoder)
│   ├── lightning_CVAE.py      # PyTorch Lightning wrapper with learnable embeddings & composite loss├── requirements.txt

│   └── perceptual_loss.py     # VGG16-based perceptual loss (relu2_2 features)
│
├── models/
│   ├── ResNet50.py            # Standard ResNet-50 (baseline)
│   ├── ResNetCBAM.py          # ResNet + Convolutional Block Attention Module
│   ├── ResNetSE.py            # ResNet + Squeeze-and-Excitation blocks
│   └── MLP.py                 # Ensemble gating network (MultiModalMetaLearner)
│
├── model_confs/               # YAML configuration files for each model
│   ├── CVAE.yaml
│   ├── ResNet50.yaml
│   ├── ResNetCBAM.yaml
│   ├── ResNetSE.yaml
│   └── MLP.yaml
│
├── CheXpert-v1.0-small/       # Dataset (download from Kaggle)
│   ├── train.csv
│   ├── train_balanced.csv     # Oversampled/CVAE-augmented training CSV
│   ├── valid.csv
│   ├── train/
│   └── valid/
│
├── experiments/               # Training outputs (checkpoints, logs)
│   ├── cvae/                  # CVAE experiments per policy
│   └── U-Ones/                # Classification model experiments of U-Ones policy
|   |__ U-Zeros/               # Classification model experiments of U-Zeros policy
│
└── EDA/                       # Exploratory data analysis notebooks/outputs
```

---

## Prerequisites

```bash
pip install -r requirements.txt
```

**Key dependencies:** `torch`, `torchvision`, `pytorch-lightning`, `pandas`, `numpy`, `scikit-learn`, `torchmetrics`, `PyYAML`,

---

## Dataset

The CheXpert dataset contains chest X-ray images with 14 multi-label observations. Labels can be **positive (1)**, **negative (0)**, or **uncertain (-1)**. Uncertain labels are handled via configurable policies during training:

| Policy       | Uncertain (-1) mapped to |
|--------------|--------------------------|
| **U-Ones**   | 1 (positive)             |
| **U-Zeros**  | 0 (negative)             |
| **U-Smooth** | 0.55                     |
| **U-Ignore** | NaN (excluded from loss) |

Images are resized to 224×224 RGB. Metadata (age, sex, view orientation) is normalized and used as auxiliary input.

---

## Data Augmentation (CVAE)

A **Conditional Variational Autoencoder** generates disease-specific synthetic X-ray images to address class imbalance.

- **Encoder:** CNN (3×224×224 → 256 channels @ 14×14) + disease/metadata embeddings → latent space (μ, log σ²)
- **Decoder:** Latent z → deconvolution upsampling → reconstructed image (3×224×224)
- **Latent dim:** 128 | **Embedding dim:** 64
- **Loss:** MSE reconstruction + SSIM (0.3) + β-KL (warmup to 0.001) + VGG16 perceptual loss (0.1)
- **Free Bits KL** (0.5 nats/dim) prevents posterior collapse

**References:**
- [Conditional VAE](https://arxiv.org/abs/1908.09008)
- [Learnable Conditional Embeddings](https://medium.com/data-science/conditional-variational-autoencoders-with-learnable-conditional-embeddings-e22ee5359a2a)

---

## Classification Models

### 1. [ResNet-50](https://arxiv.org/abs/1512.03385) (Baseline)
Standard ResNet-50 with Bottleneck blocks [3, 4, 6, 3] → 14 class logits.

### 2. [ResNet-SE](https://arxiv.org/pdf/1709.01507) (Squeeze-and-Excitation)
ResNet with SE blocks: Global avg pool → FC → ReLU → FC → Sigmoid → channel recalibration.
- [Implementation reference](https://apxml.com/courses/cnns-for-computer-vision/chapter-5-attention-transformers-vision/implementing-attention-blocks-practice)

### 3. [ResNet-CBAM](https://arxiv.org/abs/1807.06521) (Convolutional Block Attention Module)
ResNet with CBAM: channel attention (avg+max pool → shared MLP) + spatial attention (concat → conv → sigmoid).

### 4. Ensemble (MLP Gating Network)
- Concatenates predictions from all 3 models + patient metadata → $3 \times 14 + 4 = 46$ input features
- Architecture: 46 → 128 (BN, ReLU, Dropout 0.3) → 64 (BN, ReLU, Dropout 0.3) → 14 logits
- Learns which classifier to trust for specific image types

---

## PCA + Ensemble

Standard models give a single probability, but this doesn't indicate whether the model is confident or guessing. Use the probability as output and integrate this into our dataset to train a MLP. Since 14 * 3 = 42 features, so we have to add 42 dimensions to the features. In order to lower the dimension of data, we Leverge PCA to capture the max variance in the dataset. Then, use the best n_components to train our MLP.

---

## Weighted Ensemble
According to the performance of each class from each model, assgin a classification weight to each class of each model to make a final decision on a sample.


## Training & Testing

### Analyze Class Distribution

```bash
python analyze_distribution.py
```

### CVAE

**Train** (one command per uncertainty policy):
```bash
python trainCVAE.py -p U-Ones
python trainCVAE.py -p U-Zeros
python trainCVAE.py -p U-Smooth
```

**Test:**
```bash
python testCVAE.py -c CVAE_configs.yaml -ckpt experiments/cvae/CVAE_U-Ones/lightning_logs/version_1/checkpoints/<checkpoint>.ckpt
```

**Generate synthetic images:**
```bash
python generate_dataset.py --checkpoint "experiments/cvae/CVAE_U-Zeros/lightning_logs/version_0/checkpoints/<checkpoint>.ckpt"
```

### Classification Models

**Train** (specify config and policy):
```bash
python trainModels.py -c model_confs/ResNet50.yaml -p U-Ones
python trainModels.py -c model_confs/ResNetCBAM.yaml -p U-Ones
python trainModels.py -c model_confs/ResNetSE.yaml -p U-Ones
```

**Test:**
```bash
python testModels.py -c model_confs/ResNetCBAM.yaml -p U-Ones
```

Replace `-c` with any model config and `-p` with any policy (`U-Ones`, `U-Zeros`, `U-Smooth`).

### PCA Ensemble MLP

```bash
python trainMLP.py -c model_confs/MLP.yaml -p U-Ones
python testMLP.py
```

### Weighted Ensemble MLP
```bash
python weightedEnsemble.py -c model_confs/MLP.yaml -p U-Ones
python testWeightedEnsemble.py
```

---

## Pretrained Models

- [experiments](https://drive.google.com/drive/folders/1PqiQ_yJkTNa8mqyL2yFbUux5TuqPifht?usp=sharing)

Place downloaded `experiments/` which includes all the models we have trained in this project.




