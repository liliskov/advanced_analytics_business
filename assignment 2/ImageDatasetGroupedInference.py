import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import os
import torch

class ImageDatasetGroupedInference(Dataset):
    def __init__(self, data, root_dir, transform=None):
        super().__init__()
        self.data = data
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data['screenshots'].iloc[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path)
        label = self.data['sentiment'].iloc[idx]
        index = self.data['id'].iloc[idx]
        
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label), index