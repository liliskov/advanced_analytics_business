import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3,padding =1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32,32, kernel_size=3,padding =1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        self.regressor = nn.Linear(64 * 16 * 16, 1)



    def forward(self,x):
        x = self.feature_extractor(x)
        x = self.regressor(x)
        return x