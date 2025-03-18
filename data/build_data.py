from sklearn.model_selection import train_test_split
import os
import torch
from . import dataset
from osgeo import gdal

def getTif(imgPath):
    print('正在读取TIFF...')
    ds = gdal.Open(imgPath)
    GeoTrans = ds.GetGeoTransform()  # 仿射矩阵
    ref_width = ds.RasterXSize  # 栅格矩阵的列数
    ref_height = ds.RasterYSize  # 栅格矩阵的行数
    ref_Proj = ds.GetProjection()  # 投影信息
    # imgArr = ds.ReadAsArray(0, 0, ref_width, ref_height)  # 将数据写成数组，对应栅格矩阵
    return ds, GeoTrans, ref_width, ref_height, ref_Proj

class BuildData:
    def __init__(self, dataset_mode, file_path=(None,None,None), batch_size=(None,None,None), DDP=False):

        # 数据加载模式
        self.dataset_mode = dataset_mode

        # 各个数据集批大小
        self.batch_train, self.batch_val, self.batch_test = batch_size
        self.path_train, self.path_val, self.path_test = file_path
        self.DDP = DDP

    def get_dataset(self):
        """
        self.dataset_mode == 0
            正常加载模式, val和test路径为空就返回空
        
        self.dataset_mode == 1
            需要有val和test的返回值, 但是没有val和test的数据集, 需要从train里面分一部出来
            注意, 分出来的三者是没有交集的
            采用sklearn.model_selection.train_test_split划分配对的路径

        self.dataset_mode == 2
            需要有val和test的返回值, 但是没有val和test的数据集, 需要从train里面分一部出来
            注意, 分出来的三者是有交集的(test和val是train的子集)
            采用sklearn.model_selection.train_test_split划分配对的路径
        """
        if self.dataset_mode == 0:
            print("mode 0 加载模式")
            assert self.path_train is not None, "训练数据集路径为空, Message from data.build_data"
            assert self.path_val is not None, "验证数据集路径为空, Message from data.build_data"
            assert self.path_test is not None, "测试数据集路径为空, Message from data.build_data"

            path_train = self.path_train
            path_val = self.path_val
            path_test = self.path_test

            # 训练集列表生成
            img_train = [path_train + "%06d_img" % x + ".tif" for x in range(len(os.listdir(path_train)) // 2)]
            label_train = [path_train + "%06d_lab" % x + ".tif" for x in range(len(os.listdir(path_train)) // 2)]

            # 验证集列表生成
            img_val = [path_val + "%06d_img" % x + ".tif" for x in range(len(os.listdir(path_val)) // 2)]
            label_val = [path_val + "%06d_lab" % x + ".tif" for x in range(len(os.listdir(path_val)) // 2)]

            # 测试集列表生成
            img_test = [path_test + "%06d_img" % x + ".tif" for x in range(len(os.listdir(path_test)) // 2)]
            label_test = [path_test + "%06d_lab" % x + ".tif" for x in range(len(os.listdir(path_test)) // 2)]

            # 数据加载
            if self.DDP:
                
                dataloader_train = dataset.MakeDataloaderDDP(img_train, label_train, batch_size=self.batch_train, mark="train")
                dataloader_val = dataset.MakeDataloaderDDP(img_val, label_val, batch_size=self.batch_val, mark="val")
                dataloader_test = dataset.MakeDataloaderDDP(img_test, label_test, batch_size=self.batch_test, mark="test")
            else:
                # 数据加载
                dataloader_train = dataset.MakeDataloader(img_train, label_train, batch_size=self.batch_train, mark="train")
                dataloader_val = dataset.MakeDataloader(img_val, label_val, batch_size=self.batch_val, mark="val")
                dataloader_test = dataset.MakeDataloader(img_test, label_test, batch_size=self.batch_test, mark="test")
            
            return dataloader_train, dataloader_val, dataloader_test


        if self.dataset_mode == 1:
            # mode 1
            assert self.path_train is not None, "数据集路径为空, Message from data.build_data"
            assert self.path_val is None, "数据集路径为空, Message from data.build_data"
            assert self.path_test is None, "数据集路径为空, Message from data.build_data"

            path_train = self.path_train
            # 训练 -> 验证 -> 测试
            # 训练集列表生成
            img_train = [path_train + "%06d_img" % x + ".png" for x in range(len(os.listdir(path_train)) // 2)]
            label_train = [path_train + "%06d_lab" % x + ".png" for x in range(len(os.listdir(path_train)) // 2)]

            # 拆分训练集为训练集、验证集、测试集
            img_train, img_val, label_train, label_val = train_test_split(img_train, label_train, test_size=0.3, random_state=42)
            img_val, img_test, label_val, label_test = train_test_split(img_val, label_val, test_size=0.5, random_state=42)
            
            if self.DDP:
                
                dataloader_train = dataset.MakeDataloaderDDP(img_train, label_train, batch_size=self.batch_train, mark="train")
                dataloader_val = dataset.MakeDataloaderDDP(img_val, label_val, batch_size=self.batch_val, mark="val")
                dataloader_test = dataset.MakeDataloaderDDP(img_test, label_test, batch_size=self.batch_test, mark="test")
            else:
                # 数据加载
                dataloader_train = dataset.MakeDataloader(img_train, label_train, batch_size=self.batch_train, mark="train")
                dataloader_val = dataset.MakeDataloader(img_val, label_val, batch_size=self.batch_val, mark="val")
                dataloader_test = dataset.MakeDataloader(img_test, label_test, batch_size=self.batch_test, mark="test")
            
            return dataloader_train, dataloader_val, dataloader_test

        if self.dataset_mode == 2:
            """
            self.dataset_mode == 2
            需要有val和test的返回值, 但是没有val和test的数据集, 需要从train里面分一部出来
            注意, 分出来的三者是有交集的(test和val是train的子集)
            采用sklearn.model_selection.train_test_split划分配对的路径
            """
            # mode 2
            assert self.path_train is not None, "数据集路径为空, Message from data.build_data"
            assert self.path_val is None, "数据集路径为空, Message from data.build_data"
            assert self.path_test is None, "数据集路径为空, Message from data.build_data"

            path_train = self.path_train
            # 训练 -> 验证 -> 测试
            # 训练集列表生成
            img_train = [path_train + "%06d_img" % x + ".png" for x in range(len(os.listdir(path_train)) // 2)]
            label_train = [path_train + "%06d_lab" % x + ".png" for x in range(len(os.listdir(path_train)) // 2)]

            # 拆分训练集为训练集、验证集、测试集
            _, img_val_test, _, label_val_test = train_test_split(img_train, label_train, test_size=0.3, random_state=42)
            img_val, img_test, label_val, label_test = train_test_split(img_val_test, label_val_test, test_size=0.5, random_state=42)
            
            if self.DDP:
                
                dataloader_train = dataset.MakeDataloaderDDP(img_train, label_train, batch_size=self.batch_train, mark="train")
                dataloader_val = dataset.MakeDataloaderDDP(img_val, label_val, batch_size=self.batch_val, mark="val")
                dataloader_test = dataset.MakeDataloaderDDP(img_test, label_test, batch_size=self.batch_test, mark="test")
            else:
                # 数据加载
                dataloader_train = dataset.MakeDataloader(img_train, label_train, batch_size=self.batch_train, mark="train")
                dataloader_val = dataset.MakeDataloader(img_val, label_val, batch_size=self.batch_val, mark="val")
                dataloader_test = dataset.MakeDataloader(img_test, label_test, batch_size=self.batch_test, mark="test")
            
            return dataloader_train, dataloader_val, dataloader_test

    @staticmethod
    def open_txt(path_txt):  # 逐行读取txt文件
        with open(path_txt, 'r') as f:
            data_txt = []
            line = f.readline().strip('\n')
            while line:
                data_txt.append(line)
                line = f.readline().strip('\n')
        return data_txt


if __name__ == '__main__':

    pass
