import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchmetrics import Precision, Recall
import torch
from torch.utils.data import DataLoader
from ImageDataset import ImageDataset
from Make_Price_train_test import Make_Price_train_test
from Net import Net
from tqdm import tqdm

class PricePredicter:
    def __init__(self):
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_transforms  = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, :, :])
        ])
        test_transforms  = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, :, :]),
        ])
        price_im = Make_Price_train_test()
        train_data = ImageDataset(data = price_im.get_training_set(), root_dir='images/', transform=train_transforms)
        test_data = ImageDataset(data = price_im.get_test_set(), root_dir='images/', transform=test_transforms)
        
        self.dl_train = DataLoader(
            train_data,
            batch_size=16,
            shuffle = True
        )
        self.dl_test = DataLoader(
            test_data,
            batch_size=16,
            shuffle = True
        )
        self.train_classifier()

    def train_classifier(self):
        
        net = Net()
        net.to(self.device)

        criterion = nn.MSELoss()
        criterion.to(self.device)

        optimizer = optim.Adam(net.parameters(), lr = 0.001)

        print("Model training...")
        for epoch in range(1):
            print(f"current epoch: {epoch}")
            running_loss = 0.0
            with tqdm(self.dl_train) as t:
                for images, labels in t:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = net(images)
                    loss = criterion(outputs, labels.unsqueeze(1).float())
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    t.set_description(desc=f"Loss: {loss.item()}")

            # epoch_loss = running_loss / len(self.dl_train)
            # print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

        total_loss = 0.0
        total_samples = 0
        net.eval()
        print("Running validation...")
        with torch.no_grad():
            for images, labels in tqdm(self.dl_test):
                outputs = net(images)
                loss = criterion(outputs,labels.unsqueeze(1).float())
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
            average_loss = total_loss / total_samples
            print(f"Validation Loss: {average_loss}")

trainer = PricePredicter()