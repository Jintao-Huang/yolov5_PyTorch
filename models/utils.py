# Author: Jintao Huang
# Time: 2020-5-18

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torch.nn as nn
import torch
import cv2 as cv
from PIL import Image
import numpy as np


def freeze_layers(model, layers):
    """冻结层

    :param model:
    :param layers: List
    :return:
    """
    for name, parameter in model.named_parameters():
        for layer in layers:
            if layer in name:  # 只要含有名字即可
                parameter.requires_grad_(False)
                break
        else:
            parameter.requires_grad_(True)


class FrozenBatchNorm2d(nn.Module):
    """copy from `torchvision.ops.misc`"""

    def __init__(self, in_channels, eps=1e-5, *args, **kwargs):  # BN: num_features, eps=1e-5, momentum=0.1
        """`*args, **kwargs` to prevent error"""
        self.eps = eps
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(in_channels))
        self.register_buffer("bias", torch.zeros(in_channels))
        self.register_buffer("running_mean", torch.zeros(in_channels))
        self.register_buffer("running_var", torch.ones(in_channels))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))  # Prevent load errors

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("expected 4D input (got %dD input)" % x.dim())
        mean, var = self.running_mean, self.running_var
        weight, bias = self.weight, self.bias

        mean, var = mean[:, None, None], var[:, None, None]
        weight, bias = weight[:, None, None], bias[:, None, None]
        return (x - mean) * torch.rsqrt(var + self.eps) * weight + bias


def get_scale_pad(img_shape, new_shape, training=False, stride=32):
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    ratio = min(new_shape[0] / img_shape[0], new_shape[1] / img_shape[1])
    new_unpad = int(round(img_shape[1] * ratio)), int(round(img_shape[0] * ratio))  # new image unpad shape

    # Compute padding
    pad_w, pad_h = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if not training:  # detect
        pad_w, pad_h = pad_w % stride, pad_h % stride
    pad_w, pad_h = pad_w / 2, pad_h / 2  # divide padding into 2 sides
    return ratio, new_unpad, (pad_w, pad_h)


def resize(img, new_shape=(640, 640), training=False, color=(114, 114, 114), stride=32):
    """copy from official yolov5"""
    # Resize and pad image
    shape = img.shape[:2]  # current shape(H, W)
    ratio, new_unpad, (pad_w, pad_h) = get_scale_pad(shape, new_shape, training, stride)
    if ratio != 1:  # resize
        img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))  # 防止0.5, 0.5
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)  # add border(grey)
    return img, ratio, (pad_w, pad_h)  # 处理后的图片, 比例, padding的像素


def pil_to_cv(img):
    """转PIL.Image到cv (Turn PIL.Image to CV(BGR))

    :param img: PIL.Image. RGB. 0-255
    :return: ndarray. BGR. 0-255,
    """
    mode = img.mode
    arr = np.asarray(img)
    if mode == "RGB":
        arr = cv.cvtColor(arr, cv.COLOR_RGB2BGR)
    else:
        raise ValueError("img.mode nonsupport")
    return arr


def cv_to_pil(arr):
    """转cv到PIL.Image (Turn CV(BGR) to PIL.Image)

    :param arr: ndarray. BGR. 0-255
    :return: PIL.Image. RGB. 0-255
    """
    if arr.dtype != np.uint8:
        raise ValueError("arr.dtype nonsupport")
    arr = cv.cvtColor(arr, cv.COLOR_BGR2RGB)
    return Image.fromarray(arr)
