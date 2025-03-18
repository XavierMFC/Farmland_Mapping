# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
#
# https://github.com/bubbliiiing/segformer-pytorch/tree/master/nets
#
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5



class MultiTaskHead(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    # 输入通道是64
    """

    def __init__(self, in_channels=48, class_nums=2):
        super().__init__()
        # torch.Size([2, 128, 256, 256])
        
        
        self.distance = nn.Sequential(  # 这个卷集核的个数需要考虑下多少合适 input=64
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),

            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),

            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.GELU(),

            nn.Conv2d(in_channels//2, 1, kernel_size=1),
            nn.Sigmoid()  # dist的范围是0~1
        )

        self.bound = nn.Sequential(
            nn.Conv2d(in_channels+1, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),

            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),

            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.GELU(),

            nn.Conv2d(in_channels//2, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.seg = nn.Sequential(
            nn.Conv2d(in_channels+2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),

            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),

            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.GELU(),

            nn.Conv2d(in_channels//2, class_nums, kernel_size=1),  # 接Softmax
        )

    def forward(self, x):  # 64
        identity = x

        dists = self.distance(x)  # 64
        psp_dist = torch.cat([identity, dists], dim=1)  # 64 + 1

        # print(psp_dist.shape)
        bounds = self.bound(psp_dist)  # 65 -> 1
        dist_bound_psp = torch.cat([dists, bounds, identity], dim=1)  # 64 + 1 + 1 -> 66

        segs = self.seg(dist_bound_psp)  # 172
        return segs, bounds, dists


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim*4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)
    
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3]).contiguous()
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3]).contiguous()
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3]).contiguous()
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3]).contiguous()

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        
        return _c

class SegFormer(nn.Module):
    def __init__(self, num_classes = 21, phi = 'b0', pretrained = False):
        super(SegFormer, self).__init__()

        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]

        self.backbone   = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)

        self.embedding_dim   = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]

        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)

        self.multiHead = MultiTaskHead(in_channels=self.embedding_dim, class_nums=num_classes)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        
        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)
        
        seg, edge, dist = self.multiHead.forward(x)

        seg = F.interpolate(seg, size=(H, W), mode='bilinear', align_corners=True)
        edge = F.interpolate(edge, size=(H, W), mode='bilinear', align_corners=True)
        dist = F.interpolate(dist, size=(H, W), mode='bilinear', align_corners=True)

        return seg, edge, dist

def get_segformer_multiTWA(num_classes=2, phi="b2", pretrained=True):
    model = SegFormer(
        num_classes = num_classes,
        phi = phi,
        pretrained = pretrained,
        )
    return model

if __name__ == '__main__':
    pass