import cv2
import numpy as np
from utils import raster_processing
from processing_edge import edge
from processing_seg import seg
from utils import marph
from skimage import morphology

def marph(img):
    # 膨胀操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    img = cv2.dilate(img, kernel, iterations=2)
    # 腐蚀操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    img = cv2.erode(img, kernel, iterations=2)
    return img

def work_flow():
    rasterProcessing = raster_processing.rasterProcessing()
    # s1 process seg
    print('step 1 ...')
    seg_path = "/SAM_Various_Files/STL/dataset_1024X1024_s128_b4_0.989515/test_20cm_128_seg.tif"
    seg_ds, seg_GeoTrans, seg_width, seg_height, seg_Proj = rasterProcessing.getTif(seg_path)
    seg_Arr = seg_ds.ReadAsArray(0, 0, seg_width, seg_height)
    seg_Arr_p = marph(seg_Arr)

    # s2 process edge
    print('step 2 ...')
    edge_path = "/SAM_Various_Files/STL/dataset_1024X1024_s128_b4_0.989515/test_20cm_128_edge.tif"
    edge_ds, edge_GeoTrans, edge_width, edge_height, edge_Proj = rasterProcessing.getTif(edge_path)
    edge_Arr = edge_ds.ReadAsArray(0, 0, edge_width, edge_height)
    edge_Arr[edge_Arr < 0.005758265] = 0
    edge_Arr[edge_Arr != 0] = 1
    edge_Arr = edge_Arr_p.astype(np.uint8)

    edge_Arr_p = marph(edge_Arr)
    edge_Arr_p[edge_Arr_p != 0] = 1

    edge_Arr_p = morphology.skeletonize(edge_Arr_p)
    edge_Arr_p = edge_Arr_p.astype(np.uint8) * 255

    edge_Arr_p = cv2.dilate(edge_Arr_p, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    edge_Arr_p = edge_Arr_p.astype(np.uint8) * 255

    # s3 merge
    print('step 3 ...')
    seg_Arr_p[edge_Arr_p == True] = 0
    
    # save
    print('s4')
    results =  seg_Arr_p[np.newaxis, ...]
    rasterProcessing.save_Tif(
        results, 
        "/SAM_Various_Files/STL/dataset_1024X1024_s128_b4_0.989515/test_20cm_128_seg.tif", 
        "/SAM_Various_Files/STL/dataset_1024X1024_s128_b4_0.989515/results.tif"
        )
    # to shp

if __name__ == "__main__":
    work_flow()
    # 0.005758265