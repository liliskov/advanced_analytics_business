import torch
import torchvision.models as models
import torch.nn as nn
class EfficientNetModel:
    def __init__(self, freeze_resnet = True):
        super(EfficientNetModel, self).__init__()
        self.resnet50 = models.efficientnet_b5(weights = 'EfficientNet_B5_Weights.DEFAUL')
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
            nn.Softmax(dim = 1)
        )

    def forward(self,x):
        x = self.resnet50(x)
        # Flatten the output for the custom classifier
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _get_num_features(self):
        # Forward dummy input through the ResNet model to get number of features
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)  # Assuming input size of 224x224 RGB image
            features = self.resnet50(dummy_input)
        return features.shape[1]
