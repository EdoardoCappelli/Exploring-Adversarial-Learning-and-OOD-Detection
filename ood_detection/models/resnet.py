import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Use ResNet18 as base model
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # self.model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)

        # Replace the last fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)