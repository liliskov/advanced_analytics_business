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
from KeepImagesTogether import KeepImagesTogether
from TogetherDataset import TogetherDataset
import torch.nn.functional as F
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix


class ResNetClassifierTogether:

    def __init__(self):
        self.device =  'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sent_im = KeepImagesTogether()
        # TRANSFORMATIONS ALREADY DONE BEFOREHAND
        # train_transforms  = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation(45),
        #     transforms.ToTensor(),
        #     transforms.Lambda(lambda x: x[:3, :, :]),
        # ])
        # test_transforms  = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Lambda(lambda x: x[:3, :, :]),
        # ])
        transform  = transforms.Compose([
            transforms.ToTensor(),
        ])
 
        train_data = TogetherDataset(data = sent_im.get_training_set(), root_dir='ResNet_Input/', transform=transform)
        test_data = TogetherDataset(data = sent_im.get_test_set(), root_dir='ResNet_Input/', transform=transform)
        def custom_collate(batch):
            # Extract images and labels from the batch
            images_stacks = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            # Pad the image stacks to have the same number of images
            max_num_images = max(len(images) for images in images_stacks)
            padded_images_stacks = []
            for images in images_stacks:
                num_to_pad = max_num_images - len(images)
                padded_images = torch.cat([images, torch.zeros(num_to_pad, *images[0].shape)], dim=0)
                padded_images_stacks.append(padded_images)
            # Stack the padded image stacks along the batch dimension
            images_stacked = torch.stack(padded_images_stacks, dim=0)
            # Convert labels to a tensor
            labels_tensor = torch.tensor(labels)
            return images_stacked, labels_tensor
        self.dl_train = DataLoader(
            train_data,
            batch_size=8,
            shuffle = True,
            collate_fn=custom_collate
        )
        self.dl_test = DataLoader(
            test_data,
            batch_size=8,
            shuffle = True,
            collate_fn=custom_collate
        )
        self.train_model()

    def train_model(self):
        device =  self.device
        net = ResNetModel().to(device)

        class_weights = torch.tensor([1.51, 6.03, 6.70, 45.43], dtype=torch.float).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
        optimizer = optim.Adam(net.parameters(), lr = 0.001)
        print("Model training...")
        for epoch in range(3):
            print(f"current epoch: {epoch}")
            running_loss = 0.0
            with tqdm(self.dl_train) as t:
                for images, labels in t:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = net(images)
                    softmax_outputs = F.softmax(outputs, dim=1)
                    loss = criterion(softmax_outputs, labels.long())
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
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        recall_per_class = recall_score(all_labels, all_preds, average=None)
        precision_per_class = precision_score(all_labels, all_preds, average=None)
        f1_per_class = f1_score(all_labels, all_preds, average=None)
        average_recall = np.mean(recall_per_class)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        print("Recall per class:", recall_per_class)
        print("Precision per class:", precision_per_class)
        print("F1 Score per class:", f1_per_class)
        print("Average Recall:", average_recall)
        print("Confusion Matrix:\n", conf_matrix)



trainer = ResNetClassifierTogether()
