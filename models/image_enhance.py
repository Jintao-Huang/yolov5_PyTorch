# Author: Jintao Huang
# Date: 
import cv2 as cv
import torch


def hflip_image(image, points):
    """水平翻转图片, points(保持顺时针旋转)
    :param image: ndarray[N, H, W]. const
    :param points: shape[8]. const
    :return: Tuple[image_hflip: ndarray[N, H, W], points: shape[8]]"""

    width = image.shape[1]
    image = cv.flip(image, flipCode=1)
    points = points.clone().reshape((4, 2))  # left, top
    points[:, 0] = width - points[:, 0]  # points翻转
    points = points[[1, 0, 3, 2]]  # points顺序
    return image, points.flatten()


def vflip_image(image, points):
    """竖直翻转图片, points(保持顺时针旋转)"""

    height = image.shape[0]
    image = cv.flip(image, flipCode=0)
    points = points.clone().reshape((4, 2))  # left, top
    points[:, 1] = height - points[:, 1]  # points翻转
    points = points[[3, 2, 1, 0]]  # points顺序
    return image, points.flatten()


def rotate(image, points, rotate_code):
    """旋转图片, points(保持顺时针旋转)
    :param image: ndarray[N, H, W]. const
    :param points: shape[8]. const
    :param rotate_code:
        cv.ROTATE_90_CLOCKWISE: 0
        cv.ROTATE_180: 1
        cv.ROTATE_90_COUNTERCLOCKWISE: 2
    :return: Tuple[image_rotate: ndarray[N, H, W], points: shape[8]]"""

    if rotate_code == cv.ROTATE_90_CLOCKWISE:
        image, points = rotate_90(image, points)
    elif rotate_code == cv.ROTATE_180:
        image, points = rotate_180(image, points)
    elif rotate_code == cv.ROTATE_90_COUNTERCLOCKWISE:
        image, points = rotate_270(image, points)
    return image, points


def rotate_90(image, points):
    height, width = image.shape[:2]
    image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    points = points.clone().reshape((4, 2))  # left, top
    points = torch.stack([height - points[:, 1], points[:, 0]], dim=1)
    points = points[[3, 0, 1, 2]]  # points顺序
    return image, points.flatten()


def rotate_180(image, points):
    height, width = image.shape[:2]
    image = cv.rotate(image, cv.ROTATE_180)
    points = points.clone().reshape((4, 2))  # left, top
    points = torch.stack([width - points[:, 0], height - points[:, 1]], dim=1)
    points = points[[2, 3, 0, 1]]  # points顺序
    return image, points.flatten()


def rotate_270(image, points):
    height, width = image.shape[:2]
    image = cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)
    points = points.clone().reshape((4, 2))  # left, top
    points = torch.stack([points[:, 1], width - points[:, 0]], dim=1)
    points = points[[1, 2, 3, 0]]  # points顺序
    return image, points.flatten()
