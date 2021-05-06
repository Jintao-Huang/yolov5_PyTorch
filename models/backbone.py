# Author: Jintao Huang
# Date: 2021-4-26


import torch.nn as nn
from .common import Focus, ConvBnSiLU, C3, SPP

__all__ = ["YOLOv5Backbone"]


class YOLOv5Backbone(nn.Module):
    def __init__(self):
        super(YOLOv5Backbone, self).__init__()
        self.layer1 = Focus(3, 32, 3, 1, 1)  # 0
        self.layer2 = nn.Sequential(
            ConvBnSiLU(32, 64, 3, 2, 1, True),  # 1
            C3(64, 64, 1, True, 0.5)  # 2
        )
        self.layer3 = nn.Sequential(
            ConvBnSiLU(64, 128, 3, 2, 1, True),  # 3
            C3(128, 128, 3, True, 0.5)  # 4
        )
        self.layer4 = nn.Sequential(
            ConvBnSiLU(128, 256, 3, 2, 1, True),  # 5
            C3(256, 256, 3, True, 0.5)  # 6
        )
        self.layer5 = nn.Sequential(
            ConvBnSiLU(256, 512, 3, 2, 1, True),  # 7
            SPP(512, 512, (5, 9, 13)),  # 8
            C3(512, 512, 1, False, 0.5)  # 9
        )

    def forward(self, x):
        output = []
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        output.append(x)
        x = self.layer4(x)
        output.append(x)
        x = self.layer5(x)
        output.append(x)
        return output  # [x_3, x_4, x_5]
