from PIL import Image
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

CHEXPERT_CLASSES = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

def load_image(csv_path, data_dir=""):
    """
    Loads the CheXpert CSV file and prepares image paths.
    csv_path: Path to train.csv, valid.csv, or test.csv
    data_dir: Base directory where the "CheXpert-v1.0-small" folder is located
    """
    df = pd.read_csv(csv_path)
    
    # Fill NaN (unmentioned findings) with 0.0 (negative)
    df[CHEXPERT_CLASSES] = df[CHEXPERT_CLASSES].fillna(0.0)
    
    # Prepend data_dir to relative paths if provided
    if data_dir:
        df["Path"] = df["Path"].apply(lambda p: os.path.join(data_dir, p))
    
    return df

class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)  # avoid index issues
        self.transform = transform
        
        self.disease_cols = CHEXPERT_CLASSES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = row['Path']
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            x = self.transform(img)
        else:
            # Fallback: convert to tensor
            x = transforms.ToTensor()(img)
        
        # Meta data: age, sex, view
        age = float(row['Age']) / 100.0 if pd.notna(row['Age']) else 0.5
        sex = 1.0 if row.get('Sex') == 'Female' else 0.0
        
        # View encoding: 0=Lateral, 0.5=PA, 1.0=AP
        frontal_lateral = row.get('Frontal/Lateral', '')
        if frontal_lateral == 'Lateral':
            view = 0.0
        else:
            ap_pa = row.get('AP/PA', '')
            if ap_pa == 'PA':
                view = 0.5
            elif ap_pa == 'AP':
                view = 1.0
            else:
                view = 0.5   # default for unknown or missing
        
        meta = torch.tensor([age, sex, view], dtype=torch.float32)
        
        # Labels (14 diseases) – keep original -1, 0, 1 values
        labels = row[self.disease_cols].values.astype(np.float32)
        y = torch.tensor(labels, dtype=torch.float32)
        
        return x, meta, y