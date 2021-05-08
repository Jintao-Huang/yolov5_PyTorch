# Author: Jintao Huang
# Date: 2021-4-26
# 测试yolov5复现准确性
import torch
import time
import cv2 as cv
from models.yolov5 import YOLOv5
from models.utils import nms, convert_boxes, model_info
from models.dataset import LoadImages
from utils.utils import load_params, save_params, load_from_pickle
from utils.display import draw_target_in_image, resize_max


def test_detect():
    # pred = load_from_pickle("other/pred.pkl")
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = YOLOv5(80).to(device)
    model.state_dict()
    print(load_params(model, "weights/yolov5s_coco.pth"))
    dataset = LoadImages("./images", 640, 32)
    model.float().fuse().eval().requires_grad_(False)  #
    half = (device.type != 'cpu')
    # half = False
    if half:
        model.half()
    sum_t = 0.
    x, img0, path = dataset[0]
    x = x.to(device)
    x = x.half() if half else x.float()
    x /= 255
    if x.dim() == 3:
        x = x[None]
    model_info(model, x.shape[-2:])
    for i in range(200):
        t = time.time()
        target = model(x)[0]
        target = nms(target, 0.2, 0.45)
        if i >= 5:
            sum_t += time.time() - t
        convert_boxes(target[0], x.shape[-2:], img0.shape[:2])
        img = img0.copy()
        draw_target_in_image(img, target[0], "coco")
        img = resize_max(img, 720, 1080)
        cv.imshow("1", img)
        cv.waitKey(0)
    print(sum_t / (200 - 5))


def test_test():
    pass


if __name__ == "__main__":
    # detect
    with torch.no_grad():
        test_detect()
    # test
