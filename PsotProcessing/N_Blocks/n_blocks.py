'''
Author: garfield garfield@outlook.com
Date: 2025-03-15 15:12:28
LastEditors: garfield garfield@outlook.com
LastEditTime: 2025-03-16 15:50:17
FilePath: /Farmland_Mapping/PsotProcessing/N_Blocks/n_blocks.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import os
import cv2
import math
import matplotlib.pyplot as plt
from shapely import wkt, geometry
from shapely.geometry import Polygon
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import generate_binary_structure
from skimage import morphology
from skimage import io
from tqdm import tqdm


def getTiff(imgPath):
    ds = gdal.Open(imgPath)
    GeoTrans = ds.GetGeoTransform()  # 仿射矩阵
    ref_width = ds.RasterXSize  # 栅格矩阵的列数
    ref_height = ds.RasterYSize  # 栅格矩阵的行数
    ref_Proj = ds.GetProjection()  # 投影信息
    imgArr = ds.ReadAsArray(0, 0, ref_width, ref_height)  # 将数据写成数组，对应栅格矩阵
    print('Band Count:', ds.RasterCount)
    print('Band Width:', ref_width)
    print('Band Height:', ref_height)
    print('Band Projection:', ref_Proj)
    return ds, GeoTrans, ref_width, ref_height, ref_Proj, imgArr


def arr2tiff(ref, save_path, arr, dtype='uint8'):
    # plt.imsave(save_path, arr)  # 保存为jpg格式
    _, GeoTrans, ref_width, ref_height, ref_Proj, _ = getTiff(ref)
    # 保存为TIF格式
    driver = gdal.GetDriverByName("GTiff")

    # assert (ref_width, ref_height) == (arr.shape[1], arr.shape[2]), print('栅格尺寸与参考影像不匹配!')

    if dtype == 'float32':
        datasetnew = driver.Create(save_path, ref_width, ref_height, 1, gdal.GDT_Float32)
    if dtype == 'uint8':
        datasetnew = driver.Create(save_path, ref_width, ref_height, 1, gdal.GDT_Byte)

    datasetnew.SetGeoTransform(GeoTrans)
    datasetnew.SetProjection(ref_Proj)
    band = datasetnew.GetRasterBand(1)
    band.WriteArray(arr)
    datasetnew.FlushCache()  # Write to disk.必须清除缓存
    print('finished!')

def softmax_n_blocks(imgpath):
    ds, GeoTrans, ref_width, ref_height, ref_Proj, Arr = getTiff(imgpath)

    Arr = np.argmax(Arr, axis=0)  # 标准操作
    print(Arr.shape)
    
    arr2tiff(
        ref = imgpath, 
        save_path = imgpath[:-4]+'_softmax.tif', 
        arr = Arr,
        dtype='uint8'
        )

if __name__ == "__main__":
    # 单图矢量化
    imgpath = "/parcel_15cm/post_processing/merge_test_resample_15cm_Predict_seg_1024_16*16_Block.tif"
    softmax_n_blocks(imgpath)
    print('finished !')
