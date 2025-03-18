'''
Author: garfield garfield@outlook.com
Date: 2025-03-15 15:12:28
LastEditors: garfield garfield@outlook.com
LastEditTime: 2025-03-16 15:51:27
FilePath: /Farmland_Mapping/PsotProcessing/workflow_v1.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import numpy as np
from utils import raster_processing
from processing_edge import edge
from processing_seg import seg
from utils import marph


def work_flow():
    rasterProcessing = raster_processing.rasterProcessing()
    # s1 process edfe
    print('s1')
    edge_results = edge.binary(
        edgePath="/SAM_Various_Files/STL/dataset_1024X1024_s128_b4_0.960426/test_20cm_128_edge.tif",
        savePath="/SAM_Various_Files/STL/dataset_1024X1024_s128_b4_0.960426/tmp/test_20cm_128_edge_binary.tif",
        int8=None,
        thresh=0.005341493,
        marph=True,
        )
    # s2 process seg
    print('s2')
    mask_edge = seg.binary(edgePath="/SAM_Various_Files/STL/dataset_1024X1024_s128_b4_0.960426/test_20cm_128_seg.tif", 
               savePath="/SAM_Various_Files/STL/dataset_1024X1024_s128_b4_0.960426/tmp/test_20cm_128_seg_edge_binary.tif", 
               marph=True,
               )
    print('s3')

    # s3 merge edge and mask_edge
    mask_merge = edge_results + mask_edge
    
    # s4 final results
    print('s4')
    results = marph.marph(mask_merge, ks=7, skeletonize=True)
    results =  results[np.newaxis, ...]
    rasterProcessing.save_Tif(
        results, 
        "/SAM_Various_Files/STL/dataset_1024X1024_s128_b4_0.960426/test_20cm_128_seg.tif", 
        "/SAM_Various_Files/STL/dataset_1024X1024_s128_b4_0.960426/tmp/test_20cm_128_results.tif"
        )
    # to shp

if __name__ == "__main__":
    work_flow()
    #0.005758265