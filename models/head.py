# Author: Jintao Huang
# Date: 2021-4-27

import torch.nn as nn
import torch

__all__ = ["Head"]


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()

    def forward(self, x):  # e.g. List[shape[1, 128, 80, 80], shape[1, 256, 40, 40], shape[1, 512, 20, 20]]
        pass
