import torch
import torchvision.models as models
import torch.nn as nn
class ResNetModel(nn.Module):
    def __init__(self, freeze_resnet = True):
        super(ResNetModel, self).__init__()
        self.resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.resnet50 = torch.nn.Sequential(*(list(self.resnet50.children())[:-1]))
        if freeze_resnet:
            for param in self.resnet50.parameters():
                param.requires_grad = False
        num_features = self._get_num_features()
    
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 9), 
        )

    def forward(self,x):
        # if x.dim()>4:
        # iterate through the frames and return max_output
        res = [] 
        for i in range(x.shape[1]):
            y = x[:,i,:,:,:]
            y = y.view(-1, x.shape[2], x.shape[3], x.shape[4])
            y = self.resnet50(y)
            y = y.view(y.size(0), -1)
            res.append(self.classifier(y))
        mean_output = torch.stack(res, dim=0).mean(dim=0)
        return mean_output
        # x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        # x = self.resnet50(x)
        # # Flatten the output for the custom classifier
        # x = x.view(x.size(0), -1)
        # for i in frames:
            
        
        # x = self.classifier(x)
        # return x
    
    def _get_num_features(self):
        # Forward dummy input through the ResNet model to get number of features
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)  # Assuming input size of 224x224 RGB image
            features = self.resnet50(dummy_input)
        return features.shape[1]
    