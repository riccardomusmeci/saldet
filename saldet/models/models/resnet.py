import torch.nn as nn
from timm import create_model
from torch import functional as F


class ResNet(nn.Module):
    def __init__(self, model_name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.model = create_model(model_name, pretrained=pretrained, num_classes=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out1 = self.relu(self.model.bn1(self.model.conv1(x)))
        out1 = self.model.maxpool(out1)
        out2 = self.model.layer1(out1)
        out3 = self.model.layer2(out2)
        out4 = self.model.layer3(out3)
        out5 = self.model.layer4(out4)

        return out2, out3, out4, out5
