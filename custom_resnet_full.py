import torch
from monai.networks.nets import SEResNet50,EfficientNetBN
from collections import OrderedDict
from torch import nn


import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_ 
from torch.nn.init import xavier_uniform_

from collections import OrderedDict

#class CustomResNet(nn.Module):
#    def __init__(self):
#        super(CustomResNet, self).__init__()
#        # Load a pre-trained ResNet model from MONAI
#        self.resnet = SEResNet50(spatial_dims=2, in_channels=3, num_classes=2, pretrained=False)
#
#    def forward(self, x):
#        x = self.resnet(x)
#        return x
class CustomEfficientNetSimp(nn.Module):
    def __init__(self, pretrained=False, weights_path=None, freeze=False):
        super(CustomEfficientNetSimp, self).__init__()
        # Load a pre-trained EfficientNet model from MONAI
        self.efficientnet = EfficientNetBN("efficientnet-b5", in_channels=3, num_classes=1000, pretrained=False)

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
        #self.efficientnet._swish = nn.Identity()

        # Freeze the parameters of the EfficientNet model if freeze is True
        if freeze:
            for param in self.efficientnet.parameters():
                param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.2),
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
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(500, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.3),
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
   
class DualEfficientNet(nn.Module):
    def __init__(self, pretrained=False, weights_path=None, freeze=False):
        super(DualEfficientNet, self).__init__()
        # Load two pre-trained EfficientNet models from MONAI
        self.efficientnet1 = EfficientNetBN("efficientnet-b0", in_channels=3, num_classes=1000, pretrained=False)
        self.efficientnet2 = EfficientNetBN("efficientnet-b0", in_channels=3, num_classes=1000, pretrained=False)

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

            self.efficientnet1.load_state_dict(new_state_dict, strict=False)
            self.efficientnet2.load_state_dict(new_state_dict, strict=False)
        # Remove the last linear layer
        self.efficientnet1._fc = nn.Identity()
        self.efficientnet2._fc = nn.Identity()

        # Freeze the parameters of the EfficientNet models if freeze is True
        if freeze:
            for param in self.efficientnet1.parameters():
                param.requires_grad = False
            for param in self.efficientnet2.parameters():
                param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(2560, 500),  # Input size is doubled because we concatenate the outputs of two EfficientNets
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

    def forward(self, x1, x2):
        x1 = self.efficientnet1(x1)
        x2 = self.efficientnet2(x2)
        x = torch.cat((x1, x2), dim=1)  # Concatenate the outputs of the two EfficientNets
        x = self.fc(x)
        return x
    
class AttentionBlock(nn.Module):
    def __init__(self, in_features, normalize=True):
        super(AttentionBlock, self).__init__()
        
        self.normalize = normalize
        self.op = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, in_features), 
        )

    def forward(self, x1, x2):
        if self.normalize:
            x1 = torch.nn.functional.normalize(x1, p=2, dim=1)
            x2 = torch.nn.functional.normalize(x2, p=2, dim=1)

        x2_attn = self.op(x2)
        attention = torch.bmm(x2_attn, x1.unsqueeze(2))
        attention = torch.softmax(attention, dim=1)
        attention = attention.squeeze(2)
        
        out = x2 * attention
        
        return out

class DualAttentionEfficientNet(nn.Module):
    def __init__(self, pretrained=False, weights_path=None, freeze=False):
        super(DualEfficientNet, self).__init__()
        # Load two pre-trained EfficientNet models from MONAI
        self.efficientnet1 = EfficientNetBN("efficientnet-b0", in_channels=3, num_classes=1000, pretrained=False)
        self.efficientnet2 = EfficientNetBN("efficientnet-b0", in_channels=3, num_classes=1000, pretrained=False)
        self.attention = AttentionBlock(1280)  # Assuming the output size of EfficientNet is 1280
        
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

            self.efficientnet1.load_state_dict(new_state_dict, strict=False)
            self.efficientnet2.load_state_dict(new_state_dict, strict=False)
        # Remove the last linear layer
        self.efficientnet1._fc = nn.Identity()
        self.efficientnet2._fc = nn.Identity()

        # Freeze the parameters of the EfficientNet models if freeze is True
        if freeze:
            for param in self.efficientnet1.parameters():
                param.requires_grad = False
            for param in self.efficientnet2.parameters():
                param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(2560, 500),  # Input size is doubled because we concatenate the outputs of two EfficientNets
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

    def forward(self, x1, x2):
        x1 = self.efficientnet1(x1)
        x2 = self.efficientnet2(x2)
        x = self.attention(x1, x2)  # Apply attention before concatenation
        x = torch.cat((x1, x), dim=1)  # Concatenate the outputs of the two EfficientNets
        x = self.fc(x)
        return x
    