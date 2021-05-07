# Author: Jintao Huang
# Date: 2021-4-26
# 测试yolov5复现准确性
import torch
import time
import cv2 as cv
from models.yolov5 import YOLOv5
from models.utils import nms, convert_boxes, model_info
from models.dataset import LoadImages
from utils.utils import load_params, save_params
from utils.display import draw_target_in_image, resize_max

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# detect
with torch.no_grad():
    model = YOLOv5(80).to(device)
    model.state_dict()
    print(load_params(model, "weights/yolov5s.pth"))
    dataset = LoadImages("./images", 640, 32)
    model.float().fuse().eval().requires_grad_(False)
    model_info(model)
    for x, img0 in dataset:
        t = time.time()
        target = model(x.to(device))[0]
        target = nms(target, 0.2, 0.45)
        print("%.6f" % (time.time() - t))
        convert_boxes(target[0], x.shape[-2:], img0.shape[:2])
        draw_target_in_image(img0, target[0])
        img0 = resize_max(img0, 720, 1080)
        cv.imshow("1", img0)
        cv.waitKey(0)
