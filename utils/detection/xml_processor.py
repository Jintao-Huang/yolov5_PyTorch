# Author: Jintao Huang
# Time: 2020-5-24

import os
import numpy as np
import cv2 as cv
from ..display import imread, draw_target_in_image, resize_max, coco_labels, voc_labels
import xml.etree.ElementTree as ET
from models.utils import ltrb2cxcywh, cxcywh2ltrb
from copy import deepcopy

class XMLProcessor:
    """$"""

    def __init__(self, imgs_dir, annos_dir, labels="voc", imgs_set_path=None, verbose=True):
        """

        :param imgs_dir: str
        :param annos_dir: str
        :param labels: str["coco", "voc"] or list[str] or Dict[str: int] 可多对一
        :param imgs_set_path: str. txt文件, 内含图片的集合
        """
        self.imgs_dir = imgs_dir
        self.annos_dir = annos_dir
        self.imgs_set_path = imgs_set_path
        if labels == "voc":
            labels = voc_labels
        elif labels == "coco":
            labels = coco_labels
        if isinstance(labels, list):
            labels = dict(zip(labels, range(len(labels))))
        self.labels_str2int = labels
        self.labels_int2str = dict(zip(labels.values(), labels.keys()))  # 只有v < 0 忽略
        self.img_path_list = []  # List[str]
        # List[ndarray[X, 5], Tuple[W, H]]. [cls, *xywh]
        self.target_list = []
        self.verbose = verbose

    def create_labels_cache(self):
        """解析xmls.

        :return: None. 缓存见self.img_path_list, self.target_list
        """
        print("create labels cache...")
        img_path_list = []  # List[str]
        target_list = []  # List[ndarray[X, 5], Tuple[W, H]]. [cls, *xywh]
        if self.imgs_set_path:
            with open(self.imgs_set_path, "r") as f:
                annos_fname_list = ["%s.xml" % x.rstrip('\n') for x in f]
        else:
            annos_fname_list = os.listdir(self.annos_dir)
        annos_path_list = [os.path.join(self.annos_dir, fname) for fname in annos_fname_list]
        for i, annos_path in enumerate(annos_path_list):
            img_path, target = self._get_data_from_xml(annos_path)
            img_path_list.append(os.path.join(self.imgs_dir, img_path))
            target_list.append(target)
        self.img_path_list = img_path_list
        self.target_list = target_list

    def _get_data_from_xml(self, anno_path):
        """get img_path, target from xml.
        检查: (图片文件存在, 每张图片至少一个目标, 目标名在labels中.)

        :param anno_path: str
        :return: Tuple[img_path, target: List[ndarray[X, 5], Tuple[W, H]]. [cls, *xywh]]"""
        # 1. 获取文件名
        img_path = os.path.join(self.imgs_dir, os.path.basename(anno_path).replace(".xml", ".jpg"))
        # 2. 检测图片存在
        img = cv.imread(img_path)
        w, h = img.shape[1], img.shape[0]
        assert img is not None, "image not found. path: %s" % img_path
        # 3. 获取ann数据
        with open(anno_path, "r", encoding="utf-8") as f:
            text = f.read()
        # [cls, *xywh]
        xml_tree = ET.parse(anno_path)
        data_list = list(zip(
            xml_tree.findall(".//object/name"),
            xml_tree.findall(".//object/bndbox/xmin"),
            xml_tree.findall(".//object/bndbox/ymin"),
            xml_tree.findall(".//object/bndbox/xmax"),
            xml_tree.findall(".//object/bndbox/ymax"),
        ))
        if len(data_list) == 0:  # 没有框
            print("| no target. but we still put it in. path: %s" % img_path) if self.verbose else None
        # 4. 处理数据
        target_list = []  # len(NUMi, 4), len(NUMi)
        for obj_name, left, top, right, bottom in data_list:
            label = self.labels_str2int.get(obj_name.text)  # obj_name 是否存在于labels中. label: int
            if label is None:  # 目标名不在labels中
                print("`%s` not in labels. path: %s" % (obj_name, anno_path)) if self.verbose else None
                continue
            if label == -1:
                continue
            target_list.append([label, int(left.text), int(top.text), int(right.text), int(bottom.text)])

        # 5. 数据类型转换. target归一化
        target = np.array(target_list, dtype=np.float32)  # [X, 4]
        target[:, 1:] = ltrb2cxcywh(target[:, 1:])
        target[:, 1::2] /= w  # lr
        target[:, 2::2] /= h  # tb
        target = [target, (w, h)]
        return img_path, target

    def test_dataset(self):
        """测试pickle文件(输出总图片数、各个分类的目标数).

        :return: None
        """
        print("-------------------------------------------------")
        print("imgs数量: %d" % len(self.img_path_list))
        print("targets数量: %d" % len(self.target_list))
        # 获取target各个类的数目
        # 1. 初始化classes_num_dict
        classes_num_dict = {label_name: 0 for label_name in self.labels_int2str.values()}
        # 2. 累加
        for target in self.target_list:  # 遍历每一张图片
            for label in target[0][:, 0]:
                classes_num_dict[self.labels_int2str[int(label)]] += 1
        # 3. 打印
        print("classes_num:")
        for obj_name, value in classes_num_dict.items():
            print("\t%s: %d" % (obj_name, value))
        print("\tAll: %d" % sum(classes_num_dict.values()), flush=True)

    def show_dataset(self, random=False, colors=None):
        """展示数据集，一张张展示

        :param random: bool
        :param colors: Dict / List
        :return: None
        """
        target_list = deepcopy(self.target_list)
        if random:
            orders = np.random.permutation(range(len(self.img_path_list)))
        else:
            orders = range(len(self.img_path_list))
        for i in orders:  # 随机打乱
            # 1. 数据结构
            img_path = self.img_path_list[i]
            img_fname = os.path.basename(img_path)
            target, (w, h) = target_list[i]
            labels = target[:, 0]
            boxes = target[:, 1:]
            boxes = cxcywh2ltrb(boxes)
            boxes[:, 0::2] *= w  # lr
            boxes[:, 1::2] *= h  # tb
            # 2. 打开图片
            img = imread(img_path)
            draw_target_in_image(img, boxes, labels, None, self.labels_int2str, colors)
            img = resize_max(img, 720, 1080)
            cv.imshow("%s" % img_fname, img)
            cv.waitKey(0)
            cv.destroyWindow("%s" % img_fname)
