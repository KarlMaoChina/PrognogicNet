import torch
from monai.networks.nets import SEResNet50,EfficientNetBN
from collections import OrderedDict
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
    def __init__(self, pretrained=False, weights_path=None):
        super(CustomEfficientNet, self).__init__()
        # Load a pre-trained EfficientNet model from MONAI
        self.efficientnet = EfficientNetBN("efficientnet-b0", in_channels=3, num_classes=1000, pretrained=False)

        if pretrained and weights_path is not None:
            pretrained_dict = torch.load(weights_path)
            print(pretrained_dict.keys())

            new_state_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                name = k
                if "_blocks." in k:
                    parts = k.split(".")
                    name = ".".join(parts[:-1]) + ".0." + parts[-1]
                new_state_dict[name] = v

            self.efficientnet.load_state_dict(new_state_dict, strict=False)

        self.fc = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.fc(x)
        return x
    

