'''
Author: garfield && rsipy@outlook.com
Date: 2023-07-02 11:20:54
LastEditors: garfield garfield@outlook.com
LastEditTime: 2025-03-16 15:50:14
FilePath: /code/rsipy/evaluation/IoU_FullTif.py

Description: 计算大图预测结果与真值的IoU

Copyright (c) 2023 by rsipy@outlook.com, All Rights Reserved. 
'''
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



def binary(edgePath, savePath, int8=None, thresh=None, marph=True):
    """ Convert Edge result to skeleton

    Args:
        edgePath (str): edge result, which is GeoTIFF format, abs path
        savePath (str): full abs save path
        int8 (str): The type of convert prob Edge to binary map via max-min to int8 first
        thresh: via specified number to convert it to binary map
        marph (bool): whether appling marph processing

    Return:
        None

    """

    # 读取预测文件和真值文件
    src_mask = gdal.Open(edgePath)  # 读取头文件
    mask_H, mask_W, mask_Band = src_mask.RasterYSize, src_mask.RasterXSize, src_mask.RasterCount  # RasterYSize行数, RasterXSize列数
    mask = src_mask.ReadAsArray()  # [0, 1]
    assert (int8 != None or thresh == None) or (int8 == None or thresh != None), print('int method error !')
    if int8:
        # int8 量化
        mask = (mask - mask.min())/ (mask.max() - mask.min()) * 255
    
        # 阈值分割 & 二值化
        mask[mask < 0] = 0  # 最小值为0
        mask[mask > 255] = 255  # 最小值为0
        mask = mask.astype(np.uint16)

        ret, _ = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret = ret*0.6
        mask[mask < ret] = 0
        mask[mask >= ret] = 255
    if thresh:
        mask[mask < thresh] = 0
        mask[mask >= thresh] = 255
        mask = mask.astype(np.uint16)
    if marph:
        # # 膨胀操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.dilate(mask, kernel, iterations=2)
        # # 腐蚀操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.erode(mask, kernel, iterations=2)

    # 提取骨干
    mask[mask != 0] = 1
    mask = morphology.skeletonize(mask)
    mask = mask.astype(np.uint8) * 255
    
    # 膨胀骨干
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    mask = mask.astype(np.uint8) * 255
    

            
    # 保存
    driver_mask = gdal.GetDriverByName('GTiff')
    out_mask = driver_mask.Create(savePath, mask_W, mask_H, 1, gdal.GDT_Byte)  # driver.Create(resultname, 行, 列, band, gdal.GDT_Float32)
    out_mask.SetGeoTransform(src_mask.GetGeoTransform())
    out_mask.SetProjection(src_mask.GetProjection())

    out_mask.GetRasterBand(1).WriteArray(mask, 0, 0)

    out_mask.FlushCache()
    del out_mask

    print("finished")
    return mask


if __name__ == "__main__":
    binary(edgePath, savePath, int8=None, thresh=0.005341493, marph=True)
    edgePath = "/parcel_15cm/post_processing/processingEdge/merge_test_resample_15cm_Predict_slidingwindow_256_edge.tif"
    savePath = "/parcel_15cm/post_processing/processingEdge/merge_test_resample_15cm_Predict_slidingwindow_256_edge_int8_ostu_skeleton.tif"

    binary(edgePath, savePath)