# Author: Jintao Huang
# Date: 2021-4-27
import torch.nn as nn
from .backbone import YOLOv5Backbone
from .fpn import PANet
from .head import Head
from .loss import Loss
from .dataset import post_process, pre_process
from .anchor_generator import AnchorGenerator

__all__ = ["YOLOv5"]


class YOLOv5(nn.Module):
    def __init__(self):
        super(YOLOv5, self).__init__()
        self.backbone = YOLOv5Backbone()
        self.fpn = PANet()
        self.head = Head()
        self.anchor_gen = AnchorGenerator()
        self.loss = Loss()

    def forward(self, x, target=None, img0_shape=None):
        """detect时, N = 1.

        :param x: Tensor[N, 3, H, W]. e.g. shape(1, 3, 640, 640)
        :param target: (N, 9). train
        :param img0_shape: Tuple. detect
        :return: loss: shape[] or target[9]
        """
        img_shape = x.shape[2:]
        x = self.backbone(x)  # e.g. List[shape[1, 128, 80, 80], shape[1, 256, 40, 40], shape[1, 512, 20, 20]]
        x = self.fpn(x)  # e.g. List[shape[1, 128, 80, 80], shape[1, 256, 40, 40], shape[1, 512, 20, 20]]
        x = self.head(x)
        anchors = self.anchor_gen(x.device)
        if target is not None:  # train
            x = self.loss(x, target, anchors)
        else:  # detect
            assert x.shape[0] == 1, "batch_size != 1"
            x = post_process(x[0], anchors, img0_shape, img_shape)
        return x
