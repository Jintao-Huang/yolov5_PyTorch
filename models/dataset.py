# Author: Jintao Huang
# Date:
import torch.utils.data as tud
import torch
import cv2 as cv
from .utils import resize_pad, get_scale_pad
import os


class LoadImages:
    def __init__(self, image_path, image_size=640, stride=32):
        """

        :param image_path: str. dir or file
        """
        if os.path.isdir(image_path):
            self.image_path_list = [os.path.join(image_path, image_fname) for image_fname in os.listdir(image_path)]
        else:
            self.image_path_list = [image_path]
        self.image_size = image_size
        self.stride = stride

    def __getitem__(self, idx):
        """

        :param idx:
        :return: x: Tensor[1, 3, H, W], img0: ndarray[H0, W0, 3]
        """
        image_path = self.image_path_list[idx]
        img0 = cv.imread(image_path)
        x = resize_pad(img0, self.image_size)[0]
        x = x[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to (C, H, W))
        x = torch.as_tensor(x / 255, dtype=torch.float32)[None]
        return x, img0

    def __len__(self):
        return len(self.image_path_list)
