import torch
from monai.networks.nets import SEResNet50,EfficientNetBN
from collections import OrderedDict
from torch import nn

#class CustomResNet(nn.Module):
#    def __init__(self):
#        super(CustomResNet, self).__init__()
#        # Load a pre-trained ResNet model from MONAI
#        self.resnet = SEResNet50(spatial_dims=2, in_channels=3, num_classes=2, pretrained=False)
#
#    def forward(self, x):
#        x = self.resnet(x)
#        return x
    

class CustomEfficientNet(nn.Module):
    def __init__(self, pretrained=False, weights_path=None, freeze=False):
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
        # Remove the last linear layer
        self.efficientnet._fc = nn.Identity()

        # Freeze the parameters of the EfficientNet model if freeze is True
        if freeze:
            for param in self.efficientnet.parameters():
                param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(1280, 500),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 2)
        )
        
        self.pretrained = pretrained
        self.weights_path = weights_path
        self.freeze = freeze

    def get_model_info(self):
        return {
            "pretrained": self.pretrained,
            "weights_path": self.weights_path,
            "freeze": self.freeze
        }

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.fc(x)
        return x
    
class StdEfficientNet(nn.Module):
    def __init__(self):
        super(StdEfficientNet, self).__init__()
        # Load an EfficientNet model from MONAI
        self.efficientnet = EfficientNetBN("efficientnet-b0", in_channels=3, num_classes=2, pretrained=False)
    


    def get_model_info(self):
        return {
            "pretrained": False,
            "weights_path": None,
            "freeze": False
        }
    def forward(self, x):
        x = self.efficientnet(x)
        return x