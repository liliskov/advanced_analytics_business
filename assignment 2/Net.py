import torch.nn as nn
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3,padding =1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64,64, kernel_size=3,padding =1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            # nn.Conv2d(64,64, kernel_size=3,padding =1),
            # nn.BatchNorm2d(64),
            # nn.ELU(),
            nn.Flatten()
        )
        self.classification = nn.Linear(2359296, num_classes)



    def forward(self,x):
        x = self.feature_extractor(x)
        x = self.classification(x)
        return x