"""
1. 高分辨率尺寸很大
2. 读写时间很长
3. 因此尽可能减少读写次数
4. 采用边读边写
"""

"""
Created on Thu Nov  4 14:28:10 2021
@author: garfield
Function: 读入一张TIFF, 输出结果
"""
import sys
import os


from collections import OrderedDict
import shutil
import numpy as np
import os
from imageio import imread
from tqdm import tqdm
from osgeo import gdal
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import math
from model import segformer
from matplotlib import pyplot as plt
from collections import OrderedDict
warnings.filterwarnings("ignore")

def preprocessing(img):
    # mean=[[[123.91]], [[127.88]], [[97.69]]], std=[[[41.4]], [[34.79]], [[41.21]]]
    # mean=[[[123.91]], [[127.88]], [[97.69]]]
    # std=[[[41.4]], [[34.79]], [[41.21]]]
    # img = img.astype(np.float32)
    # img -= mean
    # img /= std
    img = img.astype("float")
    # 归一化
    img = img / img.max()
    return img


def predict_sliding_window(imgPath, savePath, model, device, *args):
    """
    按照间隔预测大tiff
    """
    print("imgPath = ", imgPath)
    print("savePath = ", savePath)
    # 裁剪参数
    patchH = args[0]
    patchW = args[1]
    discardH = args[2]
    discardW = args[3]
    midSizeH = patchH - 2 * discardH
    midSizeW = patchW - 2 * discardW
    assert midSizeH > 0 and midSizeW > 0, '间隔为负数, midSize设置不正确'

    # 读取文件
    src = gdal.Open(imgPath)  # 读取头文件
    imgH, imgW, imgBand = src.RasterYSize, src.RasterXSize, src.RasterCount  # RasterYSize行数, RasterXSize列数
    print("imgBand:", imgBand)

    # 创建保存的头文件 Seg
    driver_seg = gdal.GetDriverByName('GTiff')
    out_seg = driver_seg.Create(savePath[0], imgW, imgH, 1, gdal.GDT_Byte)  # driver.Create(resultname, 行, 列, band, gdal.GDT_Float32)
    out_seg.SetGeoTransform(src.GetGeoTransform())
    out_seg.SetProjection(src.GetProjection())

    # 创建保存的头文件 edge
    driver_seg = gdal.GetDriverByName('GTiff')
    out_edge = driver_seg.Create(savePath[1], imgW, imgH, 1, gdal.GDT_Float32)  # driver.Create(resultname, 行, 列, band, gdal.GDT_Float32)
    out_edge.SetGeoTransform(src.GetGeoTransform())
    out_edge.SetProjection(src.GetProjection())

    """
    裁剪方式：
    不补零
    两头的直接原始的，中间的才取预测的中间值
    """
    # 理论上的尺寸 discardH + n * midSizeH = imgH - discardH
    nH = math.ceil((imgH - 2 * discardH) / midSizeH)  # 理论上循环的尺寸
    nW = math.ceil((imgW - 2 * discardW) / midSizeW)  # 理论上循环的尺寸
    
    model.eval()
    for i in tqdm(range(nH)):
        for j in tqdm(range(nW)):
            startH = i * midSizeH  # 起始位置
            startW = j * midSizeW  # 起始位置

            if i == nH - 1:
                startH = imgH - patchH  # 起始位置
            if j == nW - 1:
                startW = imgW - patchW  # 起始位置

            crop = src.ReadAsArray(startW, startH, patchW, patchH)  # 取出原始影像
            crop = preprocessing(crop)
            # 预测
            crop_1_p = torch.from_numpy(crop).type(torch.FloatTensor)
            crop_1_p = crop_1_p.to(device)
            crop_1_p = torch.unsqueeze(crop_1_p, 0)

            with torch.no_grad():

                pre_seg, pre_edge, pre_dist = model(crop_1_p)

                pre_seg = pre_seg.argmax(dim=1)
                pre_seg = pre_seg.cpu()[0, :, :]
                pre_edge = pre_edge.cpu()[0, 0, :, :]

            # 去除需要的部分，写入影像
            writeH = startH
            writeW = startW
            if i != 0 and i != nH - 1:  # 等于的时候直接
                writeH = startH + discardH
                pre_seg = pre_seg[discardH:discardH + midSizeH, :]
                pre_edge = pre_edge[discardH:discardH + midSizeH, :]

            if j != 0 and j != nW - 1:
                writeW = startW + discardW
                pre_seg = pre_seg[:, discardW:discardW + midSizeW]
                pre_edge = pre_edge[:, discardW:discardW + midSizeW]

            out_seg.GetRasterBand(1).WriteArray(pre_seg.numpy(), writeW, writeH)
            out_edge.GetRasterBand(1).WriteArray(pre_edge.numpy(), writeW, writeH)
            # refresh
            out_seg.FlushCache()
            out_edge.FlushCache()

    del out_seg


    # 后处理
    # vectorlization_v2.raster2shp(savePath, shpPath, (17, 17), 32)
    print("finished")



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # device = "cpu"
    "路径及文件保存设置"
    savepath = [["/code/Farmland_Mapping/runs/inference_result/test_20cm_128_seg.tif", "/code/Farmland_Mapping/runs/inference_result/test_20cm_128_edge.tif"]]  # 根目录
    
    imgpath = ["/code/Farmland_Mapping/dataset/full_tiff/s3_LinearP1.tif"]  # 图像路径

    "模型及权重加载"
    modelpath = "/code/Farmland_Mapping/runs/weight/dataset_1024X1024_s128_b4_0.983011.pt"

    # 网络
    model = segformer.get_segformer_multiTWA(
        num_classes=2,
        phi="b1",
        pretrained=False,
        )
    model = model.to(device)

    multi_GPU = False

    if multi_GPU:
        state_dict = torch.load(modelpath, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7::]  # remove 'module' if DataParallel was used
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        state_dict = torch.load(modelpath, map_location=device)
        model.load_state_dict(state_dict)
    
    for i in range(len(savepath)):
        predict_sliding_window(imgpath[i], savepath[i], model, device, *(1024, 1024, 128, 128))
    print("finished")
