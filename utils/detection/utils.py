# Author: Jintao Huang
# Time: 2020-5-18
import torchvision.transforms.functional as transF
import torch
import torchvision.transforms as trans


def hflip_image(image, target):
    """水平翻转图片, target
    :param image: PIL.Image. const
    :param target: Dict. const
    :return: image_hflip: PIL.Image, target: Dict"""

    image = transF.hflip(image)
    width = image.width
    boxes = target["boxes"].clone()  # ltrb
    boxes[:, 0], boxes[:, 2] = width - boxes[:, 2], width - boxes[:, 0]
    labels = target['labels'].clone()
    return image, {"boxes": boxes, "labels": labels}

