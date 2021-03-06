# Author: Jintao Huang
# Time: 2020-6-6

import torch
import torchvision.transforms.transforms as trans
from PIL import Image
from ..display import pil_to_cv, cv_to_pil, draw_target_in_image, imwrite, resize_max
import os
import cv2 as cv
import time


class Predictor:
    """$"""

    def __init__(self, model, device, colors=None, labels=None):
        """

        :param model:
        :param device:
        :param colors: List.
        :param labels: List
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.colors = colors or default_colors
        self.labels = labels or coco_labels

    def pred(self, x, score_thresh=0.5, nms_thresh=0.5):
        """

        :param x: Tensor[1, C, H, W]
        :param score_thresh:
        :param nms_thresh:
        :return: target: Dict
        """
        with torch.no_grad():
            x = self.model(x)
            x = nms(x, 0.2, 0.45)
        return target

    # def _pred_image(self, dataset, image_size="max", score_thresh=0.5, nms_thresh=0.5):
    #     """
    #
    #     :param dataset: LoadImages
    #     :param image_size: None / int / "max"
    #     :param score_thresh: float
    #     :param nms_thresh: float
    #     :return: image: ndarray[H, W, C]
    #     """
    #
    #     target = self.pred(image, image_size, score_thresh, nms_thresh)
    #     draw_target_in_image(image0, target, self.colors, self.labels)
    #     return image_o
    #
    # def pred_image_and_show(self, image_path, image_size="max", score_thresh=0.5, nms_thresh=0.5):
    #     with Image.open(image_path) as image:
    #         image = self._pred_image(image, image_size, score_thresh, nms_thresh)
    #     image_show = resize_max(image, 720, 1080)
    #     cv.imshow("predictor", image_show)
    #     cv.waitKey(0)
    #
    # @staticmethod
    # def _default_out_path(in_path):
    #     """
    #
    #     :param in_path: str
    #     :return: return out_path
    #     """
    #     path_split = in_path.rsplit('.', 1)
    #     return path_split[0] + "_out." + path_split[1]
    #
    # def pred_image_and_save(self, image_path, image_out_path=None, image_size="max", score_thresh=0.5, nms_thresh=0.5):
    #     image_out_path = image_out_path or self._default_out_path(image_path)
    #     with Image.open(image_path) as image:
    #         image = self._pred_image(image, image_size, score_thresh, nms_thresh)
    #     imwrite(image, image_out_path)
    #
    # def pred_video_and_save(self, video_path, video_out_path=None, image_size="max", score_thresh=0.5, nms_thresh=0.5,
    #                         exist_ok=False, show_on_time=False):
    #     video_out_path = video_out_path or self._default_out_path(video_path)
    #     if not exist_ok and os.path.exists(video_out_path):
    #         raise FileExistsError("%s is exists" % video_out_path)
    #
    #     cap = cv.VideoCapture(video_path)
    #     out = cv.VideoWriter(video_out_path, cv.VideoWriter_fourcc(*"mp4v"), int(cap.get(cv.CAP_PROP_FPS)),
    #                          (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
    #     assert cap.isOpened()
    #     fps = cap.get(cv.CAP_PROP_FPS)
    #     frame_num = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    #     print("????????????: %s" % fps)
    #     self.model.eval()
    #     self._pred_video_now = True
    #     for i in range(frame_num):
    #         ret, image = cap.read()  # BGR
    #         if ret is False:
    #             break  # ?????????
    #         start = time.time()
    #         image = cv_to_pil(image)  # -> PIL.Image
    #         image = self._pred_image(image, image_size, score_thresh, nms_thresh)
    #         print("\r>> %d / %d. ????????????: %f" % (i + 1, frame_num, time.time() - start), end="", flush=True)
    #         if show_on_time:
    #             image_show = resize_max(image, 720, 1080)
    #             cv.imshow("video", image_show)
    #             if cv.waitKey(1) in (ord('q'), ord('Q')):
    #                 exit(0)
    #         out.write(image)
    #     print()
    #     cap.release()
    #     out.release()
    #     self._pred_video_now = False
