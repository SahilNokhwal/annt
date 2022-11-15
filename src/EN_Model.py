import torch.nn as nn
import torch
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

class EN_Model(nn.Module):
    def __init__(self, num_classes:int):
        super().__init__()
        # self.network = models.efficientnet_b0(pretrained=True)
        # override_params["image_size"]=(32, 32)
        self.network = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes) #in_size=(32,32)
        # num_ftrs = self.network.fc.in_features
        # self.network.fc = nn.Linear(num_ftrs, num_classes)
                
    def forward(self, input):
        return self.network(input)
    
    