# ------------------------------------------------------------------------------
# https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/dataloaders/custom_transforms.py
# ------------------------------------------------------------------------------

import torch
import random
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter

class Standardliza(object):
    """
    Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = img.astype(np.float32)
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}

class Normalize(object):
    """
    Normalize a tensor image with min and max.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = img.astype(np.float32)
        img = img/255

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    swap color axis because
    numpy image: H x W x C
    torch image: C X H X W
    """

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = img.astype(float)
        mask = np.array(mask).astype(float)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if len(mask.shape) == 2:
            mask = mask[np.newaxis, ...]
            if random.random() <= 0.5:
                img = np.flip(img, 1)
                mask = np.flip(mask, 1)
            mask = mask[0, ...]
        else:
            if random.random() <= 0.5:
                img = np.flip(img, 1)
                mask = np.flip(mask, 1)
        return {'image': img,
                'label': mask}

class ReverseRot90(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if len(mask.shape) == 2:
            mask = mask[np.newaxis, ...]
            if random.random() <= 0.5:
                img = np.rot90(img, 1 , (1, 2))
                mask = np.rot90(mask, 1 , (1, 2))
            mask = mask[0, ...]
        else:
            if random.random() <= 0.5:
                img = np.rot90(img, 1 , (1, 2))
                mask = np.rot90(mask, 1 , (1, 2))
        return {'image': img,
                'label': mask}

class RandomRotate(object):
    """
    输入图像是opencv格式
    在[-degree, degree]之间随机生成一个数
    """

    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() <= 0.4:
            img = Image.fromarray(np.uint8(img))
            mask = Image.fromarray(np.uint8(mask))
            rotate_degree = random.uniform(-1 * self.degree, self.degree)
            img = img.rotate(rotate_degree, Image.BILINEAR)
            mask = mask.rotate(rotate_degree, Image.NEAREST)
            img = np.array(img)
            mask = np.array(mask)
        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        img = Image.fromarray(np.uint8(img))
        mask = Image.fromarray(np.uint8(mask))

        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        img = np.array(img)
        mask = np.array(mask)
        return {'image': img,
                'label': mask}


class Brightness(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        p = random.uniform(0.9, 1.1)
        if random.random() < 0.3:
            for i in range(0, img.shape[2]):
                tmp = img[:, :, i].copy()
                tmp = (tmp * p).astype("uint8")
                tmp[tmp > 255] = 255
                tmp[tmp < 0] = 0
                img[:, :, i] = tmp
        return {'image': img,
                'label': mask}


class BrightnessPerPixels(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        if random.random() < 0.5:
            for i in range(0, img.shape[0]):
                for j in range(0, img.shape[1]):
                    p = random.uniform(0.9, 1.1)
                    img[i, j, 0] = np.clip(int(img[i, j, 0] * p), a_max=255, a_min=0)
                    img[i, j, 1] = np.clip(int(img[i, j, 1] * p), a_max=255, a_min=0)
                    img[i, j, 2] = np.clip(int(img[i, j, 2] * p), a_max=255, a_min=0)

        return {'image': img,
                'label': mask}
class LinearStrench(object):
    def __call__(self, sample, p=1):
        img = sample['image']
        mask = sample['label']
        for i in range(img.shape[2]):
            band = img[..., i].astype(np.float)
            # 去掉首尾
            minValue, maxValue = np.percentile(band,[p, 100-p])
            band[band < minValue] = minValue
            band[band > maxValue] = maxValue
            img[..., i] = (255 * (band - minValue)/(maxValue - minValue)).astype(np.uint8)  # CWH 0 ~ 1
        return {'image': img,
                'label': mask}

class RandomScaleCrop(object):
    """
    base_size: 输入图像的大小
    crop_size: 裁剪后的大小
    """

    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        if random.random() <= 0.5:
            # random scale (short edge)
            short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2))
            w, h, c = img.shape
            if h > w:
                ow = short_size
                oh = int(1.0 * h * ow / w)
            else:
                oh = short_size
                ow = int(1.0 * w * oh / h)
            img_new = np.zeros((ow, oh, c))

            # 图像重采样
            for j in range(c):
                img_new[:, :, j] = cv2.resize(img[:, :, j], (ow, oh), interpolation=cv2.INTER_CUBIC)
            img = img_new
            mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)

            # 补零、裁剪
            if short_size < self.crop_size:
                padh = self.crop_size - oh if oh < self.crop_size else 0
                padw = self.crop_size - ow if ow < self.crop_size else 0
                img = np.pad(img, ((math.floor(padw / 2), math.ceil(padw / 2)), (math.ceil(padh / 2), math.floor(padh / 2)), (0, 0)),
                             'constant', constant_values=(0, 0))
                mask = np.pad(mask, ((math.floor(padw / 2), math.ceil(padw / 2)), (math.ceil(padh / 2), math.floor(padh / 2))),
                              'constant', constant_values=(self.fill, self.fill))
            # random crop crop_size
            w, h, _ = img.shape
            x1 = random.randint(0, w - self.crop_size)
            y1 = random.randint(0, h - self.crop_size)
            img = img[x1:(x1 + self.crop_size), y1:(y1 + self.crop_size), :]
            mask = mask[x1:(x1 + self.crop_size), y1:(y1 + self.crop_size)]
        return {'image': img, 'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}

class Jitcolor(object):
    def __init__(self, p=0):
        self.s = random.uniform(0.5, 1.5)
        self.v = random.uniform(0.5, 1.5)
        self.p = p

    def __call__(self, sample):

        """
            img:  WHC 归一化到0-1
            s: 0 - 1
            v: 0 - 1
        Returns: 0 - 255
        """
        img = sample['image']
        mask = sample['label']

        if random.uniform(0, 1) < self.p:
            img1 = img[0:3, ...].transpose(1, 2, 0)
            img2 = img[3:6, ...].transpose(1, 2, 0)
            if random.uniform(0, 1) < self.p:
                img1 =  img1 / 255.  # 0 - 1
                img_hsv = plt.cm.colors.rgb_to_hsv(img1)

                img_hsv[:,:,1] *= self.s
                img_hsv[:,:,2] *= self.v

                img1 = plt.cm.colors.hsv_to_rgb(img_hsv)

                # convert to 0 - 255
                img1 = np.clip(img1 * 255, a_max=255, a_min=0).astype(np.uint8)
            if random.uniform(0, 1) < self.p:
                img2 = img2 / 255.
                img_hsv = plt.cm.colors.rgb_to_hsv(img2)

                img_hsv[:,:,1] *= self.s
                img_hsv[:,:,2] *= self.v

                img2 = plt.cm.colors.hsv_to_rgb(img_hsv)

                # convert to 0 - 255
                img2 = np.clip(img2 * 255, a_max=255, a_min=0).astype(np.uint8)
            
            img1 = img1.transpose(2, 0, 1)
            img2 = img2.transpose(2, 0, 1)

            img = np.concatenate((img1, img2), axis=0)  # C, H, W 将两个影像拼接起来

            return {'image': img,
                    'label': mask}
        else:
            return {'image': img,
                    'label': mask}
