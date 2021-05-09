# Author: Jintao Huang
# Date:
from torch.utils.data import Dataset
import torch
import cv2 as cv
from .utils import resize_pad, ltrb2cxcywh, cxcywh2ltrb, cxcywhn_after_pad
import os
import numpy as np


class LoadImages(Dataset):
    def __init__(self, img_path, img_size=640, stride=32):
        """

        :param img_path: str. dir or file
        """
        if os.path.isdir(img_path):
            self.img_path_list = [os.path.join(img_path, img_fname) for img_fname in os.listdir(img_path)]
        else:
            self.img_path_list = [img_path]
        self.img_size = img_size
        self.stride = stride

    def __getitem__(self, idx):
        """

        :param idx:
        :return: x: Tensor[3, H, W], img0: ndarray[H0, W0, 3], img_path: str,
        """
        img_path = self.img_path_list[idx]
        img0 = cv.imread(img_path)
        x = resize_pad(img0, self.img_size)[0]
        x = x[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to (C, H, W))
        x = np.ascontiguousarray(x)
        x = torch.from_numpy(x)
        return x, img0, img_path

    def __len__(self):
        return len(self.img_path_list)


class LoadImagesAndLabels(Dataset):
    def __init__(self, img_path_list, target_list, img_size=640, stride=32, pad=0.0, train=False, hyp=None):
        """

        :param img_path_list: List[str].
        :param target_list: List[ndarray[X, 5], Tuple[H, W]]. [cls, *xywhn]
        :param img_size: int. 640
        :param stride: int. 32
        :param pad: train: 0.0, test: 0.5
        :param train: bool
        :param hyp: Dict
            train: Dict[].
            test: Dict["batch_size"]. rect
                rect: bool. e.g. True
                batch_size: int. e.g. 16
        """
        self.img_path_list = img_path_list
        target_list, shape_list = zip(*target_list)
        # List[ndarray[X, 5]], ndarray[NUM, 2]
        self.target_list, self.shapes = list(target_list), np.array(shape_list, dtype=np.float32)
        self.img_size = img_size
        self.stride = stride
        self.pad = pad
        self.train = train
        self.hyp = hyp
        self.num_images = self.shapes.shape[0]

        if self.train:
            pass
        else:  # test: rect and batch_size
            batch_size = self.hyp['batch_size']
            self.batch_i = np.floor(np.arange(self.num_images) / batch_size).astype(np.int)
            num_batches = self.batch_i[-1] + 1

            # ratio_wh sort
            ratio_wh = self.shapes[:, 0] / self.shapes[:, 1]
            sort_i = ratio_wh.argsort()
            self.img_path_list = [self.img_path_list[i] for i in sort_i]
            self.target_list = [self.target_list[i] for i in sort_i]
            self.shapes = self.shapes[sort_i]
            ratio_wh = ratio_wh[sort_i]

            # set batch shapes
            shapes = []
            for i in range(num_batches):
                ratio_wh_i = ratio_wh[self.batch_i == i]
                ratio_min, ratio_max = np.min(ratio_wh_i), np.max(ratio_wh_i)
                if ratio_max < 1:  # w < h
                    shapes.append([ratio_max, 1])
                elif ratio_min > 1:  # w > h
                    shapes.append([1, 1 / ratio_min])
                else:
                    shapes.append([1, 1])
            # `* stride`: 需被stride整除
            self.batch_shapes = np.ceil(np.array(shapes) * self.img_size / stride + pad).astype(np.int) * stride

    def __getitem__(self, idx):
        """

        :param idx:
        :return: x: Tensor[C, H, W], target_out: Tensor[X, 6], path: str. [idx, cls, *xywhn]
        """
        if self.train:
            pass
        else:  # test: rect and batch_size
            # load img and resize
            x, _ = self._load_image(idx)
            h0, w0 = x.shape[:2]
            # Pad
            new_shape = self.batch_shapes[self.batch_i[idx]]
            x, ratio, (pad_w, pad_h) = resize_pad(x, new_shape, False, 32, True)
            img_path = self.img_path_list[idx]
            target = self.target_list[idx].copy()
            boxes = target[:, 1:]  # shape[X, 4]. cxcywhn
            cxcywhn_after_pad(boxes, w0, h0, pad_w, pad_h)
            # numpy to Tensor
            num_labels = len(target)
            target_out = torch.zeros((num_labels, 6))
            if num_labels:
                target_out[:, 1:] = torch.from_numpy(target)  # 0 代表某batch中的第几张
            x = x[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to (C, H, W)
            x = np.ascontiguousarray(x)
            x = torch.from_numpy(x)
            return x, target_out, img_path

    def __len__(self):
        return self.num_images

    @staticmethod
    def collate_fn(batch_data):
        img, target, img_path = zip(*batch_data)
        target[:, 0] = torch.arange(target)
        return torch.stack(img, 0), torch.cat(target, 0), img_path

    def _load_image(self, idx):
        """

        :param idx:
        :return: ndarray[H, W, C], ndarray[H, W]
        """
        img_path = self.img_path_list[idx]
        img0 = cv.imread(img_path)
        assert img0 is not None, 'Image Not Found ' + img_path
        img0_h, img0_w = img0.shape[:2]
        ratio = self.img_size / max(img0_h, img0_w)
        if ratio != 1:
            # new_shape = int(round(img0_w * ratio)), int(round(img0_h * ratio))
            new_shape = int(img0_w * ratio), int(img0_h * ratio)
            img = cv.resize(img0, new_shape,
                            interpolation=cv.INTER_AREA if ratio < 1 else cv.INTER_LINEAR)
                            # interpolation=cv.INTER_LINEAR)
        else:
            img = img0
        return img, img0
