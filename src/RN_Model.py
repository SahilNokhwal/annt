import torch.nn as nn
import torch
import torchvision.models as models

class RN_Model(nn.Module):
    def __init__(self, num_classes:int):
        super().__init__()
        self.network = models.resnet50(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, num_classes)
                
    def forward(self, input):
        return self.network(input)
    
    