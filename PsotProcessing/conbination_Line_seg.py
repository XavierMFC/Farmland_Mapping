'''
Author: garfield garfield@outlook.com
Date: 2025-03-15 15:12:28
LastEditors: garfield garfield@outlook.com
LastEditTime: 2025-03-16 15:51:29
FilePath: /Farmland_Mapping/PsotProcessing/conbination_Line_seg.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
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



def binary(MaskPath, LinePath, savePath):

    # 读取预测文件和真值文件
    src_mask = gdal.Open(MaskPath)  # 读取头文件
    mask_H, mask_W, mask_Band = src_mask.RasterYSize, src_mask.RasterXSize, src_mask.RasterCount  # RasterYSize行数, RasterXSize列数
    mask = src_mask.ReadAsArray()  # [0, 1]

    src_line = gdal.Open(LinePath)  # 读取头文件
    line_H, line_W, line_Band = src_line.RasterYSize, src_line.RasterXSize, src_line.RasterCount  # RasterYSize行数, RasterXSize列数
    line = src_line.ReadAsArray()  # [0, 1] unsign 1 bit

    # add line to mask
    mask[line == 1] = 0  # get rid of line

    # get every connected area using opencv
    
            
    # 保存
    driver_mask = gdal.GetDriverByName('GTiff')
    out_mask = driver_mask.Create(savePath, mask_W, mask_H, 1, gdal.GDT_Byte)  # driver.Create(resultname, 行, 列, band, gdal.GDT_Float32)
    out_mask.SetGeoTransform(src_mask.GetGeoTransform())
    out_mask.SetProjection(src_mask.GetProjection())
    
    out_mask.GetRasterBand(1).WriteArray(mask, 0, 0)

    out_mask.FlushCache()
    del out_mask

    print("finished")


if __name__ == "__main__":
    
    maskPath = "/mnt/share/Garfield/parcel_15cm/post_processing/processingEdge/merge_test_resample_15cm_Predict_slidingwindow_256_seg.tif"
    LinePath = "/mnt/share/Garfield/parcel_15cm/post_processing/processingEdge/skeleton_shp_EliminatePolygo2R.tif"
    savePath = "/mnt/share/Garfield/parcel_15cm/post_processing/processingEdge/merge_test_resample_15cm_Predict_slidingwindow_256_seg_skeleton_shp_EliminatePolygo2R.tif"

    binary(maskPath, LinePath, savePath)