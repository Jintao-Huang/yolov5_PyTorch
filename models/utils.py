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
from torch import Tensor
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


def get_scale_pad(img_shape, new_shape, auto=True, stride=32, only_pad=False):
    """

    :param img_shape: Tuple[W, H]
    :param new_shape: Tuple[W, H]
    :param auto:
    :param stride:
    :param only_pad:
    :return: ratio: float, new_unpad: Tuple[W, H], (pad_w, pad_h)
    """
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    ratio = 1 if only_pad else min(new_shape[0] / img_shape[0], new_shape[1] / img_shape[1])
    new_unpad = int(round(img_shape[0] * ratio)), int(round(img_shape[1] * ratio))  # new image unpad shape

    # Compute padding
    pad_w, pad_h = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # square
    if auto:  # detect. rect
        pad_w, pad_h = pad_w % stride, pad_h % stride
    pad_w, pad_h = pad_w / 2, pad_h / 2  # divide padding into 2 sides
    return ratio, new_unpad, (pad_w, pad_h)


def resize_pad(img, new_shape=640, auto=True, stride=32, only_pad=False, color=(114, 114, 114)):
    """copy from official yolov5 letterbox()

    :param img: ndarray[H, W, C]
    :param new_shape: Union[int, Tuple[W, H]]
    :param auto: bool. new_shape是否自动适应
    :param color: BRG
    :param stride: int
    :param only_pad: 不resize, 只pad
    :return: img: ndarray[H, W, C], ratio: float, pad: Tuple[W, H]
    """
    # Resize and pad image
    shape = img.shape[1], img.shape[0]  # Tuple[W, H]
    new_shape = (new_shape, new_shape) if isinstance(new_shape, int) else new_shape
    ratio, new_unpad, (pad_w, pad_h) = get_scale_pad(shape, new_shape, auto, stride, only_pad)
    if ratio != 1:  # resize
        img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))  # 防止0.5, 0.5
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)  # add border(grey)
    return img, ratio, (pad_w, pad_h)  # 处理后的图片, 比例, padding的像素


def ltrb2cxcywh(boxes):
    """前后要么都不归一化，要么都归一化

    :param boxes: shape[X, 4]. (left, top, right, bottom)
    :return: shape[X, 4]. (center_x, center_y, width, height)
    """
    lt, rb = boxes[:, 0:2], boxes[:, 2:4]
    wh = rb - lt
    center_xy = lt + wh / 2
    if isinstance(boxes, Tensor):
        return torch.cat([center_xy, wh], -1)
    elif isinstance(boxes, np.ndarray):
        return np.concatenate([center_xy, wh], -1)


def cxcywh2ltrb(boxes):
    """

    :param boxes: shape(X, 4). (center_x, center_y, width, height)
    :return: shape(X, 4). (left, top, right, bottom)
    """
    center_xy, wh = boxes[:, 0:2], boxes[:, 2:4]
    lt = center_xy - wh / 2
    rb = lt + wh
    if isinstance(boxes, Tensor):
        return torch.cat([lt, rb], -1)
    elif isinstance(boxes, np.ndarray):
        return np.concatenate([lt, rb], -1)


def cxcywhn_after_pad(boxes, w0, h0, pad_w, pad_h):
    """cxcywhn -> cxcywhn. pad后的调整"""
    w, h = w0 + pad_w * 2, h0 + pad_h * 2
    boxes[:, 0] = (boxes[:, 0] * w0 + pad_w) / w  # cx
    boxes[:, 1] = (boxes[:, 1] * h0 + pad_h) / h  # cy
    boxes[:, 2] = boxes[:, 2] * w0 / w  # w
    boxes[:, 3] = boxes[:, 3] * h0 / h  # h


def nms(prediction, conf_thres, iou_thres):
    """

    :param prediction: Tensor. e.g. shape[N, 15120, 25]. [cxcywh, obj_conf, cls_conf]
    :param conf_thres: float. e.g. 0.2
    :param iou_thres: float. e.g. 0.45
    :return: e.g. List[shape[62, 6]], when N = 1. [boxes_ltrb, conf, cls]. conf从大到小排序
    """
    candidates = prediction[:, :, 4] > conf_thres
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for i, x in enumerate(prediction):
        x = x[candidates[i]]  # shape[15120, 85] -> [1042, 85]
        if x.shape[0] == 0:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = cxcywh2ltrb(x[:, :4])  # shape[1042, 4]
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
    """Notice: Modify Target directly

    :param target: e.g. shape[62, 6]. [boxes_ltrb, conf, cls]
    :param img_shape: [384, 640]
    :param img0_shape: [1080, 1920]
    :return: None
    """
    boxes = target[:, :4]
    ratio = min(img_shape[0] / img0_shape[0], img_shape[1] / img0_shape[1])
    pad = (img_shape[1] - img0_shape[1] * ratio) / 2, (img_shape[0] - img0_shape[0] * ratio) / 2  # wh
    boxes[:, 0::2] -= pad[0]  # lr - w
    boxes[:, 1::2] -= pad[1]  # tb - h
    boxes.data /= ratio
    clip_boxes_to_image(boxes, img0_shape)


def fuse_conv_bn(conv, bn):
    """Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/. (freeze)

    :param conv:
    :param bn:
    :return: new_conv
    """
    fused_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                           groups=conv.groups, bias=True).requires_grad_(False).to(conv.weight.device)

    # weight
    conv_w = conv.weight.view(conv.out_channels, -1)  # [32, 12, 3, 3], [32, 108]
    # bn_out = bn_in * bn_scale + bn_b
    bn_scale = bn.weight * torch.rsqrt(bn.running_var + bn.eps)  # [32]
    bn_w = torch.diag(bn_scale)  # [32, 32]. 对角矩阵
    fused_conv.weight.data = (bn_w @ conv_w).view(fused_conv.weight.shape)

    # bias
    conv_b = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    bn_b = bn.bias - bn.running_mean * bn_scale
    fused_conv.bias.data = (bn_w @ conv_b[:, None]).flatten() + bn_b

    return fused_conv


def model_info(model, img_size=640):
    img_size = img_size if isinstance(img_size, (tuple, list)) else (img_size, img_size)
    num_params = sum(x.numel() for x in model.parameters())  # number parameters
    num_grads = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    try:  # FLOPS
        from thop import profile
        p = next(model.parameters())
        x = torch.rand((1, 3, 32, 32), dtype=p.dtype, device=p.device)
        macs, num_params = profile(model, inputs=(x,), verbose=False)
        flops = 2 * macs
        flops_str = ", %.1f GFLOPS" % (flops * img_size[0] * img_size[1] / 32 / 32 / 1e9)  # 640x640 GFLOPS
    except (ImportError, Exception):
        flops_str = ""

    print("Model Summary: %d layers, %d parameters, %d gradients%s" %
          (len(list(model.modules())), num_params, num_grads, flops_str))


def box_iou(boxes1, boxes2):
    """参考 torchvision.ops.boxes.box_iou(). for mAP.

    :param boxes1: Tensor[M, 4]
    :param boxes2: Tensor[N, 4]
    :return: Tensor[M, N]
    """

    def box_area(boxes):
        """参考 torchvision.ops.boxes.box_area()"""
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp_min(0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union
    return iou


def box_ciou(boxes1, boxes2):
    """boxes1, boxes2 计算iou时，一一对应，与torchvision中box_iou计算方式不同
    GIoU: https://arxiv.org/pdf/1902.09630.pdf
    DIoU: https://arxiv.org/pdf/1911.08287.pdf
    CIoU: (https://arxiv.org/pdf/2005.03572.pdf).
    The consistency of the aspect ratio between the anchor boxes and the target boxes is also extremely important

    :param boxes1: Tensor[X, 4]
    :param boxes2: Tensor[X, 4]
    :return: Tensor[X]"""

    def _cal_distance2(dx, dy):
        """欧式距离的平方(Euclidean distance squared)"""
        return dx ** 2 + dy ** 2

    # 1. calculate iou
    wh_boxes1 = boxes1[:, 2:] - boxes1[:, :2]  # shape[X, 2].
    wh_boxes2 = boxes2[:, 2:] - boxes2[:, :2]
    area1 = wh_boxes1[:, 0] * wh_boxes1[:, 1]  # shape[X]
    area2 = wh_boxes2[:, 0] * wh_boxes2[:, 1]

    lt_inner = torch.max(boxes1[:, :2], boxes2[:, :2])  # shape[X, 2] 内框
    rb_inner = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # shape[X, 2]
    wh_inner = (rb_inner - lt_inner).clamp_min(0)  # shape[X, 2]

    inter = wh_inner[:, 0] * wh_inner[:, 1]
    union = area1 + area2 - inter
    iou = inter / union  # [X]

    # 2. calculate 惩罚项1(中心点距离)
    lt_outer = torch.min(boxes1[:, :2], boxes2[:, :2])  # [X, 2]  外框
    rb_outer = torch.max(boxes1[:, 2:], boxes2[:, 2:])  # [X, 2]
    wh_outer = rb_outer - lt_outer  # [X, 2]
    lt_center = (boxes1[:, 2:] + boxes1[:, :2]) / 2  # [X, 2]  中心点框
    rb_center = (boxes2[:, 2:] + boxes2[:, :2]) / 2  # [X, 2]
    wh_center = lt_center - rb_center
    # dist2_outer: (外边框对角线距离的平方) The square of the diagonal distance of the outer border
    dist2_outer = _cal_distance2(wh_outer[:, 0], wh_outer[:, 1])  # [X]
    dist2_center = _cal_distance2(wh_center[:, 0], wh_center[:, 1])

    # 3. calculate 惩罚项2(aspect_ratios差距). 公式详见论文, 变量同论文
    v = (4 / np.pi ** 2) * \
        (torch.atan(wh_boxes1[:, 0] / wh_boxes1[:, 1]) -
         torch.atan(wh_boxes2[:, 0] / wh_boxes2[:, 1])) ** 2  # [X]
    with torch.no_grad():  # alpha为系数，无需梯度
        alpha = v / (1 - iou + v)  # [X]
    # diou = iou - dist2_center / dist2_outer
    ciou = iou - dist2_center / dist2_outer - alpha * v  # [X]
    return ciou
