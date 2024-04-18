import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
from torchmetrics import Precision, Recall
import torch
from torch.utils.data import DataLoader
from ImageDataset import ImageDataset
from Net import Net
from tqdm import tqdm
from Make_Sentiment_train_test import Make_Sentiment_train_test
from sklearn.metrics import recall_score
import numpy as np
import pandas as pd
from ResNetModel import ResNetModel
from EfficientNetModel import EfficientNetModel

class ClassifierEfficientNet:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sent_im = Make_Sentiment_train_test()

        train_transforms  = transforms.Compose([
            transforms.functional.resize(size = 456, interpolation ='InterpolationMode.BICUBIC'),
            transforms.functional.resized_crop(size = 456),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, :, :]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45)
        ])
        test_transforms  = transforms.Compose([
            transforms.functional.resize(size = 456, interpolation ='InterpolationMode.BICUBIC'),
            transforms.functional.resized_crop(size = 456),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, :, :]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ])
        
        train_data = ImageDataset(data = sent_im.get_training_set(), root_dir='images/', transform=train_transforms)
        test_data = ImageDataset(data = sent_im.get_test_set(), root_dir='images/', transform=test_transforms)
        self.dl_train = DataLoader(
            train_data,
            batch_size=256,
            shuffle = True
        )
        self.dl_test = DataLoader(
            test_data,
            batch_size=256,
            shuffle = True
        )
        self.train_model()

    def train_model(self):
        device =  self.device
        net = EfficientNetModel().to(device)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(net.parameters(), lr = 0.001)
        print("Model training...")
        for epoch in range(25):
            print(f"current epoch: {epoch}")
            running_loss = 0.0
            with tqdm(self.dl_train) as t:
                for images, labels in t:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = net(images)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    t.set_description(desc=f"Loss: {loss.item()}")

                
                epoch_loss = running_loss / len(self.dl_train)
                print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")  

        net.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in tqdm(self.dl_test):
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())


        recall_per_class = recall_score(all_labels, all_preds, average=None)
        average_recall = np.mean(recall_per_class)

        print("Recall per class:", recall_per_class)
        print("Average Recall:", average_recall)
        df_preds = pd.DataFrame(all_preds, columns=['predictions'])
        df_labels = pd.DataFrame(all_labels, columns=['labels'])

        # Save DataFrame to CSV
        df_preds.to_csv('preds.csv', index=False)
        df_labels.to_csv('labels.csv', index=False)

trainer = ClassifierEfficientNet()
