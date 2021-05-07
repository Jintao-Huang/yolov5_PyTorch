# Author: Jintao Huang
# Date: 2021-4-27
import torch.nn as nn
from .backbone import YOLOv5Backbone
from .fpn import PANet
from .head import Head

__all__ = ["YOLOv5"]


class YOLOv5(nn.Module):
    def __init__(self, num_classes, anchors_shape=None):
        super(YOLOv5, self).__init__()
        self.num_classes = num_classes
        self.anchors_shape = anchors_shape
        self.backbone = YOLOv5Backbone()
        self.fpn = PANet()
        self.head = Head(num_classes, anchors_shape)  # anchors_shape. 在detect时使用

    def forward(self, x):
        """detect时, N = 1.

        :param x: Tensor[N, 3, H, W]. e.g. shape(1, 3, 640, 640)
        :return: loss: shape[] or target[9]
        """
        x = self.backbone(x)  # e.g. List[shape[1, 128, 80, 80], shape[1, 256, 40, 40], shape[1, 512, 20, 20]]
        x = self.fpn(x)  # e.g. List[shape[1, 128, 80, 80], shape[1, 256, 40, 40], shape[1, 512, 20, 20]]
        x = self.head(x)
        return x
