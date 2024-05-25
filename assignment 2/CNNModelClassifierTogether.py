import torch
import torch.nn as nn
import torch.nn.init as init

class CNNModelClassifierTogether(nn.Module):
    def __init__(self):
        super(CNNModelClassifierTogether, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        num_features = self._get_num_features()
        self.classification = nn.Linear(num_features, 9)
        for layer in self.feature_extractor:
            if isinstance(layer, nn.Conv2d):
                init.xavier_uniform_(layer.weight)
    
    def forward(self,x):
        res = [] 
        for i in range(x.shape[1]):
            y = x[:,i,:,:,:]
            y = y.view(-1, x.shape[2], x.shape[3], x.shape[4])
            y = self.feature_extractor(y)
            res.append(self.classification(y))
        mean_output = torch.stack(res, dim=0).mean(dim=0)
        return mean_output
    
    def _get_num_features(self):
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.feature_extractor(dummy_input)
        return features.shape[1]
    