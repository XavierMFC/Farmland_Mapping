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

warnings.filterwarnings("ignore")

def cal_iou_tif(groundTruth_path, prediction_path, numcls, ignoreMask=None, ignoreValue=None):
    """
    param
        groundTruth: 真值的tif文件, numpy数组
        prediction: 预测的tif文件, numpy数组
        numcls: 类别数
        ignoreMask: 忽略的区域, numpy数组
        ignoreValue: 忽略的值, 默认255
    return
        iou: IoU数组
    """
    # 读取预测文件和真值文件
    src_groundTruth = gdal.Open(groundTruth_path)  # 读取头文件
    groundTruth_H, groundTruth_W, groundTruth_Band = src_groundTruth.RasterYSize, src_groundTruth.RasterXSize, src_groundTruth.RasterCount  # RasterYSize行数, RasterXSize列数

    src_prediction = gdal.Open(prediction_path)  # 读取头文件
    prediction_H, prediction_W, prediction_Band = src_prediction.RasterYSize, src_prediction.RasterXSize, src_prediction.RasterCount  # RasterYSize行数, RasterXSize列数
    
    # 读取数据
    groundTruth = src_groundTruth.ReadAsArray()
    prediction = src_prediction.ReadAsArray()

    assert groundTruth.shape == prediction.shape, "groundTruth.shape != prediction.shape"
    if ignoreMask:
        assert groundTruth.shape == ignoreMask.shape, "groundTruth.shape != ignoreMask.shape"

    # 计算IoU
    iou = np.zeros(numcls)
    for i in range(numcls):
        mask_i = groundTruth == i
        pre_i = prediction == i
        if ignoreMask:
            mask_i = mask_i & (ignoreMask != ignoreValue)
            pre_i = pre_i & (ignoreMask != ignoreValue)
        
        if ignoreValue:
            mask_i = mask_i & (groundTruth != ignoreValue)
            pre_i = pre_i & (prediction != ignoreValue)
        
        iou[i] = np.sum(mask_i & pre_i) / np.sum(mask_i | pre_i)

    return iou



def cal_iou(maskPath, prePath, numcls):

    # 读取预测文件和真值文件
    src_mask = gdal.Open(maskPath)  # 读取头文件
    mask_H, mask_W, mask_Band = src_mask.RasterYSize, src_mask.RasterXSize, src_mask.RasterCount  # RasterYSize行数, RasterXSize列数

    src_pre = gdal.Open(prePath)  # 读取头文件
    pre_H, pre_W, pre_Band = src_pre.RasterYSize, src_pre.RasterXSize, src_pre.RasterCount  # RasterYSize行数, RasterXSize列数

    assert mask_H == pre_H and mask_W == pre_W, 'mask和pre的大小不一致'

    assert mask_Band == 1 and pre_Band == 1, 'mask和pre的波段数不为1'

    # 读取数据
    mask = src_mask.ReadAsArray()
    pre = src_pre.ReadAsArray()

    # 计算IoU
    iou = np.zeros(numcls)
    for i in range(numcls):
        mask_i = mask == i
        pre_i = pre == i
        iou[i] = np.sum(mask_i & pre_i) / np.sum(mask_i | pre_i)

    return iou

if __name__ == "__main__":
    maskPath = "/parcel_15cm/T330481_202206_1128_耕地_零星_园地_minus20cm_test_15cm.tif"
    prePath = "/parcel_15cm/merge_test_resample_15cm_Predict_slidingwindow_128_seg.tif"

    numcls = 2
    iou = cal_iou(maskPath, prePath, numcls)
    print("iou = ", iou)