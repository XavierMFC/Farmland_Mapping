import torch.utils.data as data
import cv2
import torch
import numpy as np

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from PIL import Image
from . import transforms_custom as tr
# import custom_transforms as tr
from osgeo import gdal
gdal.UseExceptions()


# 传入标签和影响列表，实例化类，制作数据集
def MakeDataloader(ImgList, LabList, batch_size, mark):
    """
    常用版本
    """
    ImgList = ImgList
    LabList = LabList

    # 实例化
    datas = MakeDataset(ImgList, LabList, mark)

    Dataloaders = DataLoader(datas, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=False)
    print("%s数据  batch:%d  batch size:%d" % (mark, len(Dataloaders.dataset)//batch_size, batch_size))

    return Dataloaders

def MakeDataloaderDDP(ImgList, LabList, batch_size, mark):
    """
    分布式训练版本
    mask:管数据预处理的
    """
    ImgList = ImgList
    LabList = LabList

    # 实例化
    datas = MakeDataset(ImgList, LabList, mark)
    data_sampler = torch.utils.data.distributed.DistributedSampler(datas)
    
    #此处shuffle需要为False,可以自行在此之前先进行shuffle操作。
    Dataloaders = DataLoader(datas, batch_size=batch_size, shuffle=False, sampler=data_sampler)

    print("%s数据  batch:%d  batch size:%d" % (mark, len(Dataloaders.dataset)//batch_size, batch_size))
    return Dataloaders


# 继承torch.utils.data.Dataset类
class MakeDataset(data.Dataset):
    def __init__(self, ImgList, LabList, mark="train", type="Geotif"):
        self.ImgList = ImgList
        self.LabList = LabList
        self.mark = mark
        self.type = type
        self.mean = [[[0]], [[0]], [[0]], [[0]]]
        self.std = [[[1]], [[1]], [[1]], [[1]]]

    def __len__(self):
        return len(self.ImgList)

    def __getitem__(self, index):
        x_path = self.ImgList[index]
        y_path = self.LabList[index]

        img_x, img_y = self.read_data(x_path, y_path)  # 将数据接口写在这儿，日过有需要直接在这改

        img_y = self.multitask_label_v1(img_y)
   
        sample = {'image': img_x, 'label': img_y}

        # 务必保证 img_x 和 img_y 的通道顺序为 C * H * W
        if self.mark == "train":
            sample = self.transform_train(sample)

        if self.mark == "val":
            sample = self.transform_val(sample)

        if self.mark == "test":
            sample = self.transform_test(sample)
        
        return sample['image'], sample['label']
    
    def read_data(self, x_path, y_path):
        
        if self.type == "png":

            img_x = np.array(Image.open(x_path)).transpose(2, 0, 1)  # W, H, C -> C, H, W
            img_y = np.array(Image.open(y_path))  # W, H
            img_y[img_y == 255] = 1
            # img_y[img_y == 1] = 0
            # img_y[img_y == 2] = 1

            return img_x, img_y
        
        if self.type == "Geotif":

            img_x_src = gdal.Open(x_path)
            img_width = img_x_src.RasterXSize
            img_height = img_x_src.RasterYSize
            img_x = img_x_src.ReadAsArray(0, 0, img_width, img_height)  # 取出  C, H, W
            img_y_src = gdal.Open(y_path)
            img_y = img_y_src.ReadAsArray(0, 0, img_width, img_height)  # 取出  C, H, W

            return img_x, img_y

    def transform_train(self, sample):
        """
        根据需要选择增强方式
        # img格式: C, W, H
        # mask格式: W,H
        """
        composed_transforms = transforms.Compose([
            # tr.RandomHorizontalFlip(),
            # tr.ReverseRot90(),
            # tr.RandomScaleCrop(base_size=512, crop_size=512, fill=0),
            # tr.RandomRotate(180),
            # tr.RandomGaussianBlur(),
            # tr.BrightnessPerPixels(),
            # tr.Brightness(),
            # tr.Normalize(),
            # tr.Standardliza(mean=self.mean, std=self.std),
            tr.ToTensor()])
        return composed_transforms(sample)
    
    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            # tr.RandomHorizontalFlip(),
            # tr.RandomScaleCrop(base_size=384, crop_size=384, fill=0),

            # tr.RandomRotate(30),
            # tr.RandomGaussianBlur(),
            # tr.Normalize(),
            # tr.Standardliza(mean=self.mean, std=self.std),
            tr.ToTensor()])
        return composed_transforms(sample)
    
    def transform_test(self, sample):
        composed_transforms = transforms.Compose([
            # tr.RandomHorizontalFlip(),
            # tr.RandomScaleCrop(base_size=384, crop_size=384, fill=0),

            # tr.RandomRotate(30),
            # tr.RandomGaussianBlur(),
            # tr.Normalize(),
            # tr.Standardliza(mean=self.mean, std=self.std),
            tr.ToTensor()])
        return composed_transforms(sample)
    
    def multitask_label_v1(self, mask):
        """
        完全按照论文的步骤产生掩膜
        """
        mask = np.ascontiguousarray(mask)

        edge = cv2.Canny(mask, 0, 1)  # Canny 产生边界
        edge = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1)
        edge[edge == 255] = 1  # 将255的掩膜变成0、1二值图

        
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 0)  # 产生距离图 该距离图是经过归一化后的图像 Float
        dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

        return np.array([mask, edge, dist], dtype=float)

    def multitask_label_v2(self, mask):
        """
        对距离图取反
        """
        mask = np.ascontiguousarray(mask)

        edge = cv2.Canny(mask, 0, 1)  # Canny 产生边界
        edge = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1)
        edge[edge == 255] = 1  # 将255的掩膜变成0、1二值图

        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 0)  # 产生距离图 该距离图是经过归一化后的图像 Float
        dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        dist = 1- dist
        dist[dist==1] = 0

        return np.array([mask, edge, dist], dtype=float)





if __name__ == "__main__":
    pass