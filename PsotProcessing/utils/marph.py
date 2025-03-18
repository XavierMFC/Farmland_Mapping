
import sys
import os


from collections import OrderedDict
import math
import shutil
import numpy as np
import os
from imageio import imread
from tqdm import tqdm
from osgeo import gdal
import warnings
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology

warnings.filterwarnings("ignore")

def marph(img, ks=7, skeletonize=True):
    """ 形态学操作

    Args:
        img (_type_): 输入的二值影像
        skeletonize (bool, optional): 是否提取骨架
    """
    # 膨胀操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
    img = cv2.dilate(img, kernel, iterations=2)
    # 腐蚀操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
    img = cv2.erode(img, kernel, iterations=2)
    img = img.astype(np.uint8)*255
    if skeletonize:
        # 提取骨干
        img[img != 0] = 1
        img = morphology.skeletonize(img)
    img = img.astype(np.uint8)*255
    return img