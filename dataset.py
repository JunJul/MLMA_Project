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
    Loads the CheXpert CSV file instead of scanning directories.
    csv_path: Path to the train.csv or valid.csv
    data_dir: Base directory where the "CheXpert-v1.0-small" folder is located
    """
    df = pd.read_csv(csv_path)
    
    # 1. Fill NaN values (unmentioned diseases) with 0.0
    df[CHEXPERT_CLASSES] = df[CHEXPERT_CLASSES].fillna(0.0)
    
    if data_dir:
        df["Path"] = os.path.join(data_dir, "") + df["Path"]
        
    return df

class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df

        if transform is not None:
            self.transform = transform
        else:
            self.transform =  transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Identify the 14 disease columns
        self.disease_cols = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 
            'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 
            'Fracture', 'Support Devices'
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_path = row['Path']
        img = Image.open(img_path).convert('RGB')
        x = self.transform(img)
        
        age = float(row['Age']) / 100.0
        
        # Sex: Convert text to binary (Female = 1.0, Male = 0.0)
        sex = 1.0 if row['Sex'] == 'Female' else 0.0
        
        #  (Frontal = 1.0, Lateral = 0.0)
        if row['Frontal/Lateral'] == 'Lateral':
            view = 0.0
        elif row['AP/PA'] == 'PA':
            view = 0.5
        elif row['AP/PA'] == 'AP':
            view = 1.0
        else:
            view = 0.5
        
        meta = torch.tensor([age, sex, view], dtype=torch.float32)

        labels = row[self.disease_cols].fillna(0.0).astype(float).values
        y = torch.tensor(labels, dtype=torch.float32)

        return x, meta, y