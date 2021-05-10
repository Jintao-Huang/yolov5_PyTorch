# Author: Jintao Huang
# Date: 

from utils.detection.xml_processor import XMLProcessor
from utils.utils import load_from_pickle, save_to_pickle
import os


def make_dataset(imgs_dir, annos_dir, pkl_path, imgs_set_path, labels="voc", show_dataset=False):
    """

    :param imgs_dir: str
    :param annos_dir: str
    :param pkl_path: str
    :param imgs_set_path: str
    :param labels: str["coco", "voc"] or list[str] or Dict[str: int] 可多对一
    :param show_dataset: bool
    :return:
    """
    pkl_dir = os.path.dirname(pkl_path)
    os.makedirs(pkl_dir, exist_ok=True)
    xml_processor = XMLProcessor(imgs_dir, annos_dir, labels, imgs_set_path, True)
    if os.path.exists(pkl_path):
        img_path_list, target_list = load_from_pickle(pkl_path)
        xml_processor.img_path_list, xml_processor.target_list = img_path_list, target_list
        xml_processor.test_dataset()
    else:
        xml_processor.create_labels_cache()
        xml_processor.test_dataset()
        save_to_pickle((xml_processor.img_path_list, xml_processor.target_list), pkl_path)
        img_path_list, target_list = xml_processor.img_path_list, xml_processor.target_list
    if show_dataset:
        xml_processor.show_dataset()
    return img_path_list, target_list


if __name__ == "__main__":
    make_dataset(r"D:\datasets\VOCdevkit\VOC0712\JPEGImages",
                 r"D:\datasets\VOCdevkit\VOC0712\Annotations",
                 r"D:\datasets\VOCdevkit\VOC0712\pkl\voc_0712_test.pkl",
                 r"D:\datasets\VOCdevkit\VOC0712\ImageSets\Main\test.txt", "voc", True)
    make_dataset(r"D:\datasets\VOCdevkit\VOC0712\JPEGImages",
                 r"D:\datasets\VOCdevkit\VOC0712\Annotations",
                 r"D:\datasets\VOCdevkit\VOC0712\pkl\voc_0712_trainval.pkl",
                 r"D:\datasets\VOCdevkit\VOC0712\ImageSets\Main\trainval.txt", "voc", True)
