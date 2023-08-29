import torch
from monai.networks.nets import SEResNet50,EfficientNetBN
from torch import nn

class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        # Load a pre-trained ResNet model from MONAI
        self.resnet = SEResNet50(spatial_dims=2, in_channels=3, num_classes=2, pretrained=False)

    def forward(self, x):
        x = self.resnet(x)
        return x
    
class CustomEfficientNet(nn.Module):
    def __init__(self):
        super(CustomEfficientNet, self).__init__()
        # Load a pre-trained EfficientNet model from MONAI
        self.efficientnet = EfficientNetBN("efficientnet-b0", in_channels=3, num_classes=2, pretrained=False)

    def forward(self, x):
        x = self.efficientnet(x)
        return x
    

