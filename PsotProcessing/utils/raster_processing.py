"""
1. 线性拉伸
2. 最大最小值拉伸，可计算也可指定: 没指定时候不操作，因此必须指定最大最小
"""

from osgeo import gdal, ogr, osr
import numpy as np


class rasterProcessing(object):
    def __init__(self):
        pass

    def func_linearStrench(self, img_Path, p, save_Path):
        """

        Args:
            img_Path: (c, w, h) 文件路径, geotif
            p: 0 - 100 线性拉伸百分比

        Rturn:
            save_Path: (c, w, h) 保存路径, geotif

        """
        print('线性拉伸...')
        ds, GeoTrans, ref_width, ref_height, ref_Proj = self.getTif(img_Path)
        imgArr = ds.ReadAsArray(0, 0, ref_width, ref_height)
        imgArr_type = imgArr.dtype
        
        imgTmp = np.zeros(imgArr.shape)
        for i in range(imgArr.shape[0]):
            band = imgArr[i, ...]

            # 计算不包括背景值0的最小值和最大值
            non_zero_values = band[band != 0]
            minValue, maxValue = np.percentile(non_zero_values, [p, 100 - p])
            print("minValue=%d, maxValue=%d" % (minValue, maxValue))

            # 线性拉伸: 将背景值0保持不变
            non_zero_mask = (band != 0)
            band[non_zero_mask] = np.clip(band[non_zero_mask], minValue, maxValue)
            imgTmp[i, ...] = band
        imgTmp = imgTmp.astype(imgArr_type)
        self.save_Tif(imgTmp, img_Path, save_Path)
        print('finished !')
        return None

    def func_maxminStretch(self, img_Path, save_Path, minmax=None):
        """
        Args:
            img_Path: (c, w, h) 文件路径, geotif
            maxmin: 指定的最大最小值,与通道一致[[max, min], [max, min], [max, min], ...]

        Return:
            save_Path: (c, w, h) 保存路径, geotif

        """
        print('最大最小拉伸...')
        ds, GeoTrans, ref_width, ref_height, ref_Proj = self.getTif(img_Path)
        imgArr = ds.ReadAsArray(0, 0, ref_width, ref_height)
        imgArr_type = imgArr.dtype
        # 如果没有指定maxmin，则计算每个通道的最大和最小值
        if minmax is None:
            minmax = []
            for i in range(imgArr.shape[0]):
                channel = imgArr[i, ...]
                non_zero_values = channel[channel != 0]
                minValue, maxValue = np.percentile(non_zero_values, [0, 100])
                minmax.append([maxValue, minValue])

        # 应用最大最小拉伸到每个通道
        imgTmp = np.zeros(imgArr.shape)
        for i in range(imgArr.shape[0]):
            channel = imgArr[i, ...]
            minValue, maxValue = minmax[i]

            # 忽略背景值0
            non_zero_mask = (channel != 0)
            channel[non_zero_mask] = np.clip(channel[non_zero_mask], minValue, maxValue)
            imgTmp[i, ...] = channel
        imgTmp = imgTmp.astype(imgArr_type)
        self.save_Tif(imgTmp, img_Path, save_Path)
        print('finished !')
        return imgArr

    def getTif(self, imgPath):
        print('正在读取TIFF...')
        ds = gdal.Open(imgPath)
        GeoTrans = ds.GetGeoTransform()  # 仿射矩阵
        ref_width = ds.RasterXSize  # 栅格矩阵的列数
        ref_height = ds.RasterYSize  # 栅格矩阵的行数
        ref_Proj = ds.GetProjection()  # 投影信息
        # imgArr = ds.ReadAsArray(0, 0, ref_width, ref_height)  # 将数据写成数组，对应栅格矩阵
        return ds, GeoTrans, ref_width, ref_height, ref_Proj

    def save_Tif(self, imgArr, ref_Path, save_Path):
        """

        Args:
            imgArr: (c, h, w), 保存的栅格
            ref_Path: 参考栅格的路径，需要投影信息
            save_Path: 保存路径

        Returns:
            写入硬盘的栅格文件

        """
        print('写入栅格数据...')
        _, GeoTrans, _, _, ref_Proj = self.getTif(ref_Path)
        gdal_Type = self.get_dType(imgArr)
        driver = gdal.GetDriverByName('GTiff')
        output_dataset = driver.Create(save_Path, imgArr.shape[2], imgArr.shape[1], imgArr.shape[0], gdal_Type)
        output_dataset.SetGeoTransform(GeoTrans)
        output_dataset.SetProjection(ref_Proj)

        # 将数据写入栅格数据集的各个通道
        for band in range(1, imgArr.shape[0] + 1):
            print('writing band %d' % band)
            output_band = output_dataset.GetRasterBand(band)
            output_band.WriteArray(imgArr[band-1, ...])
            output_band.FlushCache()
            # output_dataset.GetRasterBand(band).WriteArray()
        print('finished !')
        del output_dataset
        return None

    def get_dType(self, img_Arr):
        """

        Args:
            img_Arr: 影像栅格

        Returns:
            Arr_type: 对应gdal的类型
        """
        print('读取数据类型...')
        data_type = img_Arr.dtype
        gdal_Type = None

        if data_type == np.uint8:
            gdal_Type = gdal.GDT_Byte
        elif data_type == np.uint16:
            gdal_Type = gdal.GDT_UInt16
        elif data_type == np.int16:
            gdal_Type = gdal.GDT_Int16
        elif data_type == np.uint32:
            gdal_Type = gdal.GDT_UInt32
        elif data_type == np.int32:
            gdal_Type = gdal.GDT_Int32
        elif data_type == np.float32:
            gdal_Type = gdal.GDT_Float32
        elif data_type == np.float64:
            gdal_Type = gdal.GDT_Float64

        assert gdal_Type != None, print('数据异常')
        print("数据类型:", data_type)
        return gdal_Type


if __name__ == "__main__":
    rasterProcessing = rasterProcessing()
    path = "/SAM_Various_Files/STL/TRIPLESAT_2_PSH_L4_20230410023431_00348EVI_024/sample/"
    img_set = [path + i for i in ['s3.tif']]
    for img_Path in img_set:
        save_Path = img_Path[:-4] + '_LinearP1.tif'
        rasterProcessing.func_maxminStretch(img_Path, save_Path, minmax=[[810, 1758], [594, 1502], [399, 1302], [402, 1567]])
