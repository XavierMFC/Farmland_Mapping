'''
Author: garfield && rsipy@outlook.com
Date: 2023-07-02 11:20:54
LastEditors: garfield garfield@outlook.com
LastEditTime: 2025-03-16 15:50:10
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



def binary(edgePath, savePath, marph=True):

    # 读取预测文件和真值文件
    src_mask = gdal.Open(edgePath)  # 读取头文件
    mask_H, mask_W, mask_Band = src_mask.RasterYSize, src_mask.RasterXSize, src_mask.RasterCount  # RasterYSize行数, RasterXSize列数
    mask = src_mask.ReadAsArray()  # [0, 1]
    mask = mask.astype(np.uint16)
    if marph:
        # 膨胀操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        mask_marph_d = cv2.dilate(mask, kernel, iterations=2)
        # 腐蚀操作
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        # mask = cv2.erode(mask, kernel, iterations=2)
    mask_edge = mask_marph_d - mask

    # 提取骨干
    mask_edge[mask_edge != 0] = 1
    mask_edge = morphology.skeletonize(mask_edge)
    mask_edge = mask_edge.astype(np.uint8) * 255
    # 膨胀骨干
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask_edge = cv2.dilate(mask_edge, kernel, iterations=1)
    mask_edge = mask_edge.astype(np.uint8) * 255
    
    # 判断是否存在环状结构
    

            
    # 保存
    driver_mask = gdal.GetDriverByName('GTiff')
    out_mask = driver_mask.Create(savePath, mask_W, mask_H, 1, gdal.GDT_Byte)  # driver.Create(resultname, 行, 列, band, gdal.GDT_Float32)
    out_mask.SetGeoTransform(src_mask.GetGeoTransform())
    out_mask.SetProjection(src_mask.GetProjection())

    out_mask.GetRasterBand(1).WriteArray(mask_edge, 0, 0)

    out_mask.FlushCache()
    del out_mask

    print("finished")
    return mask_edge


if __name__ == "__main__":
    binary(edgePath, savePath, int8=None, thresh=0.005341493, marph=True)
    edgePath = "/parcel_15cm/post_processing/processingEdge/merge_test_resample_15cm_Predict_slidingwindow_256_edge.tif"
    savePath = "/parcel_15cm/post_processing/processingEdge/merge_test_resample_15cm_Predict_slidingwindow_256_edge_int8_ostu_skeleton.tif"

    binary(edgePath, savePath)