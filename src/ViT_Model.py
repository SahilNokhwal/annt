import torch.nn as nn
import torch
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
# import trimm

class ViT_Model(nn.Module):
    def __init__(self, num_classes:int):
        super().__init__()
        self.network = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        # num_ftrs = self.network.fc.in_features
        # self.network.fc = nn.Linear(num_ftrs, num_classes)
                
    def forward(self, input):
        return self.network(input)
    
    