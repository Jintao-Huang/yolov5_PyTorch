# Author: Jintao Huang
# Date: 2021-4-26
# 测试yolov5复现准确性
import pickle
import torch
import cv2 as cv
from models.yolov5 import YOLOv5
from models.utils import nms, convert_boxes
from models.dataset import LoadImages
from utils.utils import load_params, save_params
from utils.display import draw_target_in_image, resize_max

with open("other/pred.pkl", "rb") as f:  # YOLOv5_official_output
    out = pickle.load(f)

# detect
with torch.no_grad():
    model = YOLOv5(80)
    print(load_params(model, "weights/yolov5s.pth"))
    dataset = LoadImages("./images", 640, 32)
    model.eval()
    for x, img0 in dataset:
        target = model(x)[0]
        target = nms(target, 0.2, 0.45)
        convert_boxes(target[0], x.shape[-2:], img0.shape[:2])
        draw_target_in_image(img0, target[0])
        img0 = resize_max(img0, 720, 1080)
        cv.imshow("1", img0)
        cv.waitKey(0)
