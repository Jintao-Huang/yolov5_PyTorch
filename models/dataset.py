# Author: Jintao Huang
# Date:
import torch.utils.data as tud
import torch
import cv2 as cv
from .utils import resize, get_scale_pad
from .image_enhance import hflip_image, vflip_image, rotate, rotate_90, rotate_180, rotate_270


class MyDataset(tud.Dataset):
    def __init__(self, image_path_list, target_list, return_img0=False, image_enhance=True):
        """

        :param image_path_list: List[str]
        :param target_list: List[torch.tensor]
        :param return_img0: False: train, True: test
        :param image_enhance: True: train, False: test
        """
        assert len(image_path_list) == len(target_list)
        self.image_path_list = image_path_list
        self.target_list = target_list
        self.return_img0 = return_img0
        self.image_enhance = image_enhance

    def __getitem__(self, idx):
        """

        :param idx:
        :return: x: Tensor[3, H, W], target: shape[9], img_ori: ndarray[H, W, 3]
        """
        image_path = self.image_path_list[idx]
        target = self.target_list[idx]  # Tensor[9]

        img0 = cv.imread(image_path)

        if self.image_enhance:
            if torch.rand(()) < 0.5:
                img0, target[1:] = hflip_image(img0, target[1:])
            if torch.rand(()) < 0.5:
                img0, target[1:] = vflip_image(img0, target[1:])
            if torch.rand(()) < 0.75:
                img0, target[1:] = rotate(img0, target[1:], torch.randint(0, 3, ()))
        x, target = pre_process(img0, target)
        if not self.return_img0:  # train
            return x, target
        else:
            return x, target, img0

    def __len__(self):
        return len(self.image_path_list)


def pre_process(x, target=None):
    """

    :param x: shape[H, W, C]. 原图. e.g. ndarray(1080, 1920, 3). BGR. 0-255
    :param target: shape[9]
    :return: x: shape[C, H, W]. e.g. Tensor(3, 640, 640). RGB. 0-1
             target: shape[9]
    """
    # 处理后的图像, 缩放比例, 宽度和高度上pad的像素数
    # e.g. shape(640, 640, 3), 0.15873015873015872, (80.0, 0.0)
    x, ratio, (pad_w, pad_h) = resize(x, 640, True, stride=32)
    if target is not None:  # train
        # 处理target
        cls = target[0:1].clone()
        reg = target[1:].clone()  # shape(8)
        reg[::2] = (reg[::2] * ratio + pad_w) / x.shape[1]  # w
        reg[1::2] = (reg[1::2] * ratio + pad_h) / x.shape[0]  # h
        target = torch.cat([cls, reg])
    # 处理image
    x = x[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to (C, H, W))
    x = torch.as_tensor(x / 255, dtype=torch.float32)

    return x, target



def post_process(x, anchor_points, img0_shape, img_shape=(640, 640)):
    """

    :param x: shape[3, 20, 20]
    :param anchor_points: [20, 20]
    :param img0_shape: Tuple
    :param img_shape: Tuple
    :return: target: {"score": cls.item(), "points": reg.reshape(4, 2)}
    """

    # 转0-1 idx
    # shape[400], shape[400, 2]
    cls, reg = torch.sigmoid(x[0]).flatten(), x[1:].permute((1, 2, 0)).reshape(-1, 2)
    pos_cls = cls >= 0.2
    points = anchor_points[pos_cls] + reg[pos_cls] / 32
    cls, indices = torch.sort(cls[pos_cls], descending=True)
    points = points[indices]
    # 转真实 idx
    ratio, _, (pad_w, pad_h) = get_scale_pad(img0_shape, (640, 640), True, 32)
    points[:, 0] = (points[:, 0] * img_shape[1] - pad_w) / ratio  # w
    points[:, 1] = (points[:, 1] * img_shape[0] - pad_h) / ratio  # h

    return {"score": cls, "points": points.reshape(-1, 2)}
