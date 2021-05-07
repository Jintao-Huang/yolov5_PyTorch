# Author: Jintao Huang
# Time: 2020-5-18

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torch.nn as nn
import torch
import cv2 as cv
from torchvision.ops.boxes import nms as _nms


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


def get_scale_pad(img_shape, new_shape, auto=True, stride=32):
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    ratio = min(new_shape[0] / img_shape[0], new_shape[1] / img_shape[1])
    new_unpad = int(round(img_shape[1] * ratio)), int(round(img_shape[0] * ratio))  # new image unpad shape

    # Compute padding
    pad_w, pad_h = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # square
    if auto:  # detect. rect
        pad_w, pad_h = pad_w % stride, pad_h % stride
    pad_w, pad_h = pad_w / 2, pad_h / 2  # divide padding into 2 sides
    return ratio, new_unpad, (pad_w, pad_h)


def resize_pad(img, new_shape=640, auto=True, color=(114, 114, 114), stride=32):
    """copy from official yolov5 letterbox()"""
    # Resize and pad image
    shape = img.shape[:2]  # current shape(H, W)
    new_shape = new_shape if isinstance(new_shape, (tuple, list)) else (new_shape, new_shape)
    ratio, new_unpad, (pad_w, pad_h) = get_scale_pad(shape, new_shape, auto, stride)
    if ratio != 1:  # resize
        img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))  # 防止0.5, 0.5
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)  # add border(grey)
    return img, ratio, (pad_w, pad_h)  # 处理后的图片, 比例, padding的像素


def ltrb_to_cxcywh(boxes):
    """

    :param boxes: shape[X, 4]. (left, top, right, bottom)
    :return: shape[X, 4]. (center_x, center_y, width, height)
    """
    lt, rb = boxes[:, 0:2], boxes[:, 2:4]
    wh = rb - lt
    center_xy = lt + wh / 2
    return torch.cat([center_xy, wh], -1)


def cxcywh_to_ltrb(boxes):
    """

    :param boxes: shape(X, 4). (center_x, center_y, width, height)
    :return: shape(X, 4). (left, top, right, bottom)
    """
    center_xy, wh = boxes[:, 0:2], boxes[:, 2:4]
    lt = center_xy - wh / 2
    rb = lt + wh
    return torch.cat([lt, rb], -1)


def nms(prediction, conf_thres, iou_thres):
    """

    :param prediction: Tensor. e.g. shape[N, 15120, 25]
    :param conf_thres: float. e.g. 0.2
    :param iou_thres: float. e.g. 0.45
    :return: e.g. List[shape[62, 6]], when N = 1. [boxes_ltrb, conf, cls]
    """
    candidates = prediction[:, :, 4] > conf_thres
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for i, x in enumerate(prediction):
        x = x[candidates[i]]  # shape[15120, 85] -> [1042, 85]
        if x.shape[0] == 0:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = cxcywh_to_ltrb(x[:, :4])  # shape[1042, 4]
        # conf: score. cls: 属于哪一个分类
        conf, cls = torch.max(x[:, 5:], 1, keepdim=True)  # shape[1042, 1], shape[1042, 1]
        x = torch.cat([box, conf, cls.float()], 1)  # [1042, 6]
        x = x[conf.flatten() > conf_thres]  # [763, 6]]
        if x.shape[0] == 0:
            continue
        boxes, scores = x[:, :4] + x[:, 5:6] * 4096, x[:, 4]
        keep_idxes = _nms(boxes, scores, iou_thres)  # NMS  # shape[62]. long
        output[i] = x[keep_idxes]
    return output


def clip_boxes_to_image(boxes, img_shape):
    # Clip boxes to image shape (height, width)
    boxes[:, 0::2].clamp_(0, img_shape[1])  # left, right
    boxes[:, 1::2].clamp_(0, img_shape[0])  # top, bottom


def convert_boxes(target, img_shape, img0_shape):
    """

    :param target: e.g. shape[62, 6]
    :param img_shape: [384, 640]
    :param img0_shape: [1080, 1920]
    :return: None
    """
    boxes = target[:, :4]
    ratio = min(img_shape[0] / img0_shape[0], img_shape[1] / img0_shape[1])
    pad = (img_shape[1] - img0_shape[1] * ratio) / 2, (img_shape[0] - img0_shape[0] * ratio) / 2  # wh
    boxes[:, 0::2] -= pad[0]  # lr - w
    boxes[:, 1::2] -= pad[1]  # tb - h
    boxes[:] /= ratio
    clip_boxes_to_image(boxes, img0_shape)
