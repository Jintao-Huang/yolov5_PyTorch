# Author: Jintao Huang
# Date: 2021-4-26
# 测试yolov5复现准确性
import torch
import time
import cv2 as cv
from models.yolov5 import YOLOv5
from models.utils import nms, convert_boxes, model_info, cxcywh2ltrb
from models.dataset import LoadImages, LoadImagesAndLabels
from utils.utils import load_params, save_params, load_from_pickle
from utils.display import draw_target_in_image, resize_max
from make_dataset import make_dataset
import numpy as np
from models.loss import Loss


def test_detect():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = YOLOv5(80).to(device)
    print(load_params(model, "weights/yolov5s_coco.pth", strict=False))
    dataset = LoadImages("./images", 640, 32)
    model.float().fuse().eval().requires_grad_(False)
    half = (device.type != 'cpu')
    # half = False
    model.half() if half else None
    sum_t = 0.
    # 预处理
    x, img0, path = dataset[0]
    x = x.to(device)
    x = x.half() if half else x.float()
    x /= 255
    x = x[None] if x.dim() == 3 else x
    model_info(model, x.shape[-2:])
    for i in range(200):
        t = time.time()
        target = model(x)[0]
        target = nms(target, 0.2, 0.45)
        if i >= 5:
            sum_t += time.time() - t
        # 后处理
        target = target[0]
        convert_boxes(target, x.shape[-2:], img0.shape[:2])
        boxes = target[:, :4].cpu().numpy()
        scores = target[:, 4].cpu().numpy()
        labels = target[:, 5].cpu().numpy()
        img = img0.copy()
        # 画图
        draw_target_in_image(img, boxes, labels, scores, "coco")
        img = resize_max(img, 720, 1080)
        cv.imshow("1", img)
        cv.waitKey(0)
    print(sum_t / (200 - 5))


def test_test():
    img_path_list, target_list = \
        make_dataset(r"D:\datasets\VOCdevkit\VOC0712\JPEGImages",
                     r"D:\datasets\VOCdevkit\VOC0712\Annotations",
                     r"D:\datasets\VOCdevkit\VOC0712\pkl\voc_0712_test.pkl",
                     r"D:\datasets\VOCdevkit\VOC0712\ImageSets\Main\test.txt", "voc", False)

    dataset = LoadImagesAndLabels(img_path_list, target_list, 640, 32, 0.5, False, {"batch_size": 1})

    # x: Tensor[C, H, W], target_out: Tensor[X, 6], path: str. [idx, cls, *xywh]
    # test show
    def test1():
        for x, target, img_path in dataset:
            print(x.shape, target, img_path)
            x = x.numpy()
            x = x.transpose(1, 2, 0)[:, :, ::-1]  # to (H, W, C), RGB to BGR,
            x = np.ascontiguousarray(x)
            h, w = x.shape[:2]
            boxes = target[:, 2:].numpy()
            labels = target[:, 1].numpy()
            boxes = cxcywh2ltrb(boxes)
            boxes[:, 0::2] *= w  # lr
            boxes[:, 1::2] *= h  # tb
            draw_target_in_image(x, boxes, labels, None, "voc")
            cv.imshow("1", x)
            cv.waitKey(0)

    # test1()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = YOLOv5(20).to(device)
    print(load_params(model, "weights/yolov5s_voc.pth", strict=False))
    model.float().fuse().eval().requires_grad_(False)
    half = (device.type != 'cpu')
    model.half() if half else None
    hyp = {"obj_pw": 0.911, "cls_pw": 0.631, "anchor_t": 2.91,
           "box_lw": 0.0296, "obj_lw": 0.301, "cls_lw": 0.06075}
    loss_func = Loss(model, hyp)
    loss = torch.zeros((4,), device=device)
    for x, target0, img_path in reversed(dataset):
        img0 = x.numpy()
        img0 = img0.transpose(1, 2, 0)[:, :, ::-1]  # to (H, W, C), RGB to BGR,
        img0 = np.ascontiguousarray(img0)
        img = img0.copy()
        # 预处理
        x, target0 = x.to(device), target0.to(device)
        x = x.half() if half else x.float()
        x /= 255
        if x.dim() == 3:
            x = x[None]
        # 预测
        target, loss_target = model(x)
        loss += loss_func([loss_t.float() for loss_t in loss_target], target0)[1]
        target = nms(target, 0.001, 0.6)

        # 后处理
        # 1
        target = target[0]
        boxes = target[:, :4].cpu().numpy()
        scores = target[:, 4].cpu().numpy()
        labels = target[:, 5].cpu().numpy()
        draw_target_in_image(img, boxes, labels, scores, "voc")
        cv.imshow("pred", img)
        # 2
        img2 = img0.copy()
        h, w = img2.shape[:2]
        boxes = target0[:, 2:].cpu().numpy()
        labels = target0[:, 1].cpu().numpy()
        boxes = cxcywh2ltrb(boxes)
        boxes[:, 0::2] *= w  # lr
        boxes[:, 1::2] *= h  # tb
        draw_target_in_image(img2, boxes, labels, None, "voc")
        cv.imshow("target", img2)
        cv.waitKey(0)


def train_test():
    pass
    # img_path_list, target_list = \
    #     make_dataset(r"D:\datasets\VOCdevkit\VOC0712\JPEGImages",
    #                  r"D:\datasets\VOCdevkit\VOC0712\Annotations",
    #                  r"D:\datasets\VOCdevkit\VOC0712\pkl\voc_0712_train.pkl",
    #                  r"D:\datasets\VOCdevkit\VOC0712\ImageSets\Main\trainval.txt", "voc", False)
    # hyp = {"hsv_h": 0.0138,
    #        "hsv_s": 0.664,
    #        "hsv_v": 0.464,
    #        "degrees": 0.373,
    #        "translate": 0.245,
    #        "scale": 0.898,
    #        "shear": 0.602,
    #        "perspective": 0.0,
    #        "flipud": 0.00856,
    #        "fliplr": 0.5,
    #        "mosaic": 1.0,
    #        "mixup": 0.243}
    # dataset = LoadImagesAndLabels(img_path_list, target_list, 640, 32, 0., True, hyp)
    #
    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    # else:
    #     device = torch.device('cpu')
    # model = YOLOv5(80).to(device)
    # print(load_params(model, "weights/yolov5s_coco.pth", strict=False))
    # model.float()
    # hyp = {"obj_pw": 0.911, "cls_pw": 0.631, "anchor_t": 2.91,
    #        "box_lw": 0.0296, "obj_lw": 0.301, "cls_lw": 0.06075}
    # loss_func = Loss(model, hyp)
    # loss = torch.zeros((4,), device=device)
    # for x, target0, img_path in reversed(dataset):
    #     img0 = x.numpy()
    #     img0 = img0.transpose(1, 2, 0)[:, :, ::-1]  # to (H, W, C), RGB to BGR,
    #     img0 = np.ascontiguousarray(img0)
    #     img = img0.copy()
    #     # 预处理
    #     x, target0 = x.to(device), target0.to(device)
    #     x /= 255
    #     if x.dim() == 3:
    #         x = x[None]
    #     # 预测
    #     target, loss_target = model(x)
    #     loss += loss_func([loss_t.float() for loss_t in loss_target], target0)[1]
    #     target = nms(target, 0.001, 0.6)
    #
    #     # 后处理
    #     # 1
    #     target = target[0]
    #     boxes = target[:, :4].cpu().numpy()
    #     scores = target[:, 4].cpu().numpy()
    #     labels = target[:, 5].cpu().numpy()
    #     draw_target_in_image(img, boxes, labels, scores, "voc")
    #     cv.imshow("pred", img)
    #     # 2
    #     img2 = img0.copy()
    #     h, w = img2.shape[:2]
    #     boxes = target0[:, 2:].numpy()
    #     labels = target0[:, 1].numpy()
    #     boxes = cxcywh2ltrb(boxes)
    #     boxes[:, 0::2] *= w  # lr
    #     boxes[:, 1::2] *= h  # tb
    #     draw_target_in_image(img2, boxes, labels, None, "voc")
    #     cv.imshow("target", img2)
    #     cv.waitKey(0)


if __name__ == "__main__":
    # detect
    with torch.no_grad():
        test_detect()
    # test
    with torch.no_grad():
        test_test()
    # train
    # train_test()
