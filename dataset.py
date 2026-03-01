from PIL import Image
import os
import pandas as pd

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.utils.data


def load_image(file_dir):
    file_path = os.listdir(file_dir)

    files = []
    labels = []

    for file in file_path:
        file_list = os.listdir(os.path.join(file_dir, file))
        files.extend(file_list)
        labels.extend([file] * len(file_list))
    
    df = pd.DataFrame({
    "file": files,
    "class": labels
    })

    df["file"] = file_dir + "/" + df["class"] + "/" + df["file"]

    return df

class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df

        # a dictionary as a label encoder
        self.encoder = self._create_label_encoder()

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
    
    # encode label
    def _create_label_encoder(self):
        unique_classes = sorted(self.df["class"].unique())
        encoder = {label: i for i, label in enumerate(unique_classes)}
        return encoder

    def __getitem__(self, index):
        img_file, label = self.df.iloc[index]
        encoded_label = self.encoder[label]

        img = Image.open(img_file).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return img, encoded_label

    def __len__(self):
        return self.df.shape[0]