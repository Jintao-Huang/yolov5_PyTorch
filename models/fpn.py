# Author: Jintao Huang
# Date: 2021-4-26
import torch.nn as nn
from .common import Concat, ConvBnSiLU, C3

__all__ = ["PANet"]


class PANet(nn.Module):
    def __init__(self):
        super(PANet, self).__init__()
        self.conv_list = nn.ModuleList([
            ConvBnSiLU(512, 256, 1, 1, 0, True),  # 10
            ConvBnSiLU(256, 128, 1, 1, 0, True),  # 14
        ])
        # 为了保持模型结构不进行合并写
        self.up_list = nn.ModuleList([
            nn.Upsample(scale_factor=2., mode="nearest"),  # 11
            nn.Upsample(scale_factor=2., mode="nearest")  # 15
        ])
        self.down_list = nn.ModuleList([
            ConvBnSiLU(128, 128, 3, 2, 1, True),  # 18
            ConvBnSiLU(256, 256, 3, 2, 1, True)  # 21
        ])
        self.cat_list = nn.ModuleList([
            Concat(1),  # 12
            Concat(1),  # 16
            Concat(1),  # 19
            Concat(1)  # 22
        ])
        self.C3_list = nn.ModuleList([
            C3(512, 256, 1, False, 0.5),  # 13
            C3(256, 128, 1, False, 0.5),  # 17
            C3(256, 256, 1, False, 0.5),  # 20
            C3(512, 512, 1, False, 0.5)  # 23
        ])

    def forward(self, x):
        x_3, x_4, x_5 = x
        x_5 = self.conv_list[0](x_5)
        x_4 = self.cat_list[0]([self.up_list[0](x_5), x_4])
        x_4 = self.conv_list[1](self.C3_list[0](x_4))
        x_3 = self.cat_list[1]([self.up_list[1](x_4), x_3])
        x_3 = self.C3_list[1](x_3)
        x_4 = self.cat_list[2]([self.down_list[0](x_3), x_4])
        x_4 = self.C3_list[2](x_4)
        x_5 = self.cat_list[3]([self.down_list[1](x_4), x_5])
        x_5 = self.C3_list[3](x_5)
        return x_3, x_4, x_5
