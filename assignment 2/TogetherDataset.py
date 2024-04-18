import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import os
import torch
from sklearn.model_selection import  train_test_split
import pandas as pd
from ast import literal_eval
import os

class TogetherDataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        super().__init__()
        self.data = data
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_names = self.data.iloc[idx, 1]
        img_names = img_names.replace('jpg', 'webp')
        img_names= literal_eval(img_names)
        img_names = [name for name in img_names if os.path.exists(os.path.join('images/', name))]
        img_names = pd.DataFrame(img_names)
        label = self.data.iloc[idx, 0]
        target_images = []
        for i in range(len(img_names)):
            img_path = os.path.join(self.root_dir, ''.join(img_names.iloc[i]))
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            target_images.append(image)
        
        stacked_images = torch.stack(target_images)
        return stacked_images, torch.tensor(label)