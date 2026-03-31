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

        self.labels = self.df[CHEXPERT_CLASSES].values

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __getitem__(self, index):
        img_file = self.df.iloc[index]['Path']
        
        label_vector = self.labels[index]
        encoded_label = torch.tensor(label_vector, dtype=torch.float32)

        # CheXpert images are grayscale, but ResNet expects 3 channels. 
        img = Image.open(img_file).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, encoded_label

    def __len__(self):
        return self.df.shape[0]