import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from ImageDataset import ImageDataset
from Net import Net
from tqdm import tqdm
from Make_Sentiment_train_test import Make_Sentiment_train_test
from sklearn.metrics import recall_score
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from ImageDatasetGroupedInference import ImageDatasetGroupedInference
import matplotlib.pyplot as plt

class SentimentClassifier:
    def __init__(self):
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_transforms  = transforms.Compose([
            # transforms.Resize((512, 288)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(45),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, :, :])
        ])
        test_transforms  = transforms.Compose([
            # transforms.Resize((512, 288)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, :, :]),
        ])
        sent_im = Make_Sentiment_train_test()
        train_data = ImageDataset(data = sent_im.get_training_set(), root_dir='ResNet_Input/', transform=train_transforms)
        test_data = ImageDatasetGroupedInference(data = sent_im.get_test_set(), root_dir='ResNet_Input/', transform=test_transforms)
        # test_data = ImageDataset(data = sent_im.get_test_set(), root_dir='ResNet_Input/', transform=test_transforms)
        class_weights = torch.tensor([1.51, 6.03, 6.70, 45.43], dtype=torch.float).to(self.device)
        sample_weights = [0]*len(train_data)
        for idx, (_, label) in enumerate(train_data):
            class_weight = class_weights[label]
            sample_weights[idx] = class_weight
        
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        
        self.dl_train = DataLoader(
            train_data,
            batch_size=8,
            # shuffle = True
            sampler=sampler
        )
        self.dl_test = DataLoader(
            test_data,
            batch_size=8
        )
        self.train_classifier()

    def train_classifier(self):
        
        net = Net(4)
        net.to(self.device)
        # weights for class imbalances: total: 6904 --> class weight = 6904/n_class
        # class_weights = torch.tensor([1.51, 6.03, 6.70, 45.43], dtype=torch.float).to(self.device)
        # criterion = nn.CrossEntropyLoss(weight=class_weights).to(self.device)
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.Adam(net.parameters(), lr = 0.001)
        train_losses = []

        print("Model training...")
        for epoch in range(3):
            print(f"current epoch: {epoch}")
            running_loss = 0.0
            with tqdm(self.dl_train) as t:
                for images, labels in t:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = net(images)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    t.set_description(desc=f"Loss: {loss.item()}")
                
                epoch_loss = running_loss / len(self.dl_train)
                train_losses.append(epoch_loss)
                print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
        # self.plot_losses(train_losses)
        
        net.eval()
        # start evaluation
        print("Model evaluating...")
        all_preds = []
        all_labels = []
        all_idx = []
        with torch.no_grad():
            for images, labels, idx in tqdm(self.dl_test):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                _, preds = torch.max(outputs, 1)
                all_idx.extend(idx.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # normal inference
        df = pd.DataFrame({'idx': all_idx, 'preds': all_preds, 'labels': all_labels})
        df.columns = ['idx', 'preds', 'labels']
        grouped = df.groupby('idx')
        grouped_preds = []
        grouped_labels = []
        for idx, group in grouped:
            majority_pred = group['preds'].mode()[0]  # Majority vote
            true_label = group['labels'].iloc[0]  # all labels in the group are the same
            grouped_preds.append(majority_pred)
            grouped_labels.append(true_label)

        # grouped inference 
        grouped_preds = np.array(grouped_preds)
        grouped_labels = np.array(grouped_labels)
        recall_per_class_grouped = recall_score(grouped_labels, grouped_preds, average=None)
        precision_per_class_grouped = precision_score(grouped_labels, grouped_preds, average=None)
        f1_per_class_grouped = f1_score(grouped_labels, grouped_preds, average=None)
        average_recall_grouped = np.mean(recall_per_class_grouped)
        conf_matrix_grouped = confusion_matrix(grouped_labels, grouped_preds)
        
        print("Recall per class, grouped inference:", recall_per_class_grouped)
        print("Precision per class, grouped inference:", precision_per_class_grouped)
        print("F1 Score per class, grouped inference:", f1_per_class_grouped)
        print("Average Recall, grouped inference:", average_recall_grouped)
        print("Confusion Matrix, grouped inference:\n", conf_matrix_grouped)

        # seperate inference
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

        

    def plot_losses(self, train_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        # plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.show()
trainer = SentimentClassifier()