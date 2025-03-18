# 读取pickle
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
import pickle
import time
from collections import deque


def polygon_area(vertices):
    """
    # 假设你有一些点的坐标
    # 注意：坐标点的顺序要组成一个封闭的多边形
    vertices = [(x1, y1), 
                (x2, y2),
                (x3, y3),
                # 添加更多的点
                ]

    # 计算多边形的面积
    area = polygon_area(vertices)

    print(f"多边形的面积为: {area}")

    """
    n = len(vertices)
    area = 0

    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]

    area = abs(area) / 2.0
    return area

def recovery_pts(pts, x_min, y_min):
    pts[:, 0] = pts[:, 0] + x_min
    pts[:, 1] = pts[:, 1] + y_min

    return pts

def toshp(ds, GeoTrans, contours, strVectorFile):
    """
    点坐标写入矢量文件
    ds :
    Geoimg :
    contours :
    strVectorFile :
    """
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")  # 为了支持中文路径
    gdal.SetConfigOption("SHAPE_ENCODING", "")  # CP936, 为了使属性表字段支持中文

    ogr.RegisterAll()  # 注册所有的驱动
    strDriverName = "ESRI Shapefile"  # 创建数据，这里以创建ESRI的shp文件为例
    oDriver = ogr.GetDriverByName(strDriverName)
    if oDriver is None:
        print("驱动不可用！")
    oDS = oDriver.CreateDataSource(strVectorFile)  # 创建数据源
    if oDS is None:
        print("创建文件失败！")

    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjectionRef())
    papszLCO = []
    oLayer = oDS.CreateLayer("Polygon", srs, ogr.wkbPolygon, papszLCO)
    if oLayer is None:
        print("图层创建失败！")
    # 创建一个叫FieldID的整型属性
    oFieldID = ogr.FieldDefn("FieldID", ogr.OFTInteger)
    oLayer.CreateField(oFieldID, 1)
    oDefn = oLayer.GetLayerDefn()

    # 生成多个多边形要素
    i = 0
    for contour in tqdm(contours):

        box1 = ogr.Geometry(ogr.wkbLinearRing)
        i += 1
        for point in contour:
            # 将像素坐标转地理坐标
            x_col = GeoTrans[0] + GeoTrans[1] * (float(point[0])) + (float(point[1])) * GeoTrans[2]
            y_row = GeoTrans[3] + GeoTrans[4] * (float(point[0])) + (float(point[1])) * GeoTrans[5]
            box1.AddPoint(x_col, y_row)
        oFeatureTriangle = ogr.Feature(oDefn)
        oFeatureTriangle.SetField(0, i)
        garden1 = ogr.Geometry(ogr.wkbPolygon)
        garden1.AddGeometry(box1)
        garden1.CloseRings()
        geomTriangle = ogr.CreateGeometryFromWkt(str(garden1))
        oFeatureTriangle.SetGeometry(geomTriangle)
        oLayer.CreateFeature(oFeatureTriangle)
    oDS.Destroy()
    xpixel = GeoTrans[1]
    ypixel = GeoTrans[5]
    return xpixel, ypixel

def getTiff(imgPath):
    ds = gdal.Open(imgPath)
    GeoTrans = ds.GetGeoTransform()  # 仿射矩阵
    ref_width = ds.RasterXSize  # 栅格矩阵的列数
    ref_height = ds.RasterYSize  # 栅格矩阵的行数
    ref_Proj = ds.GetProjection()  # 投影信息
    imgArr = ds.ReadAsArray(0, 0, ref_width, ref_height)  # 将数据写成数组，对应栅格矩阵
    print(np.unique(imgArr))
    return ds, GeoTrans, ref_width, ref_height, ref_Proj, imgArr


def getEndPoint(imgArr):
    """
    如果一个点是端点，那么要么
    args:
        imgArr : 二值化后的图像
    return:
        endPoints : 端点坐标、相应的结点
    """
    imgArr[imgArr != 0] = 1
    w, h = imgArr.shape
    endPoints = []
    for i in range(w):
        for j in range(h):
                # 零值不考虑
                if imgArr[i, j] == 0:
                    continue
                # 判断八邻域的连接情况，如果只有一个连接，那么就是端点
                # connectivity = top + below + left + right + left_top + right_top + left_below + right_below
                connectivity = int(imgArr[i-1, j]) + int(imgArr[i+1, j]) + int(imgArr[i, j-1]) + int(imgArr[i, j+1]) + int(imgArr[i-1, j-1]) + int(imgArr[i-1, j+1]) + int(imgArr[i+1, j-1]) + int(imgArr[i+1, j+1])
                if connectivity == 1:
                    # 判断是否是伪端点，通过二阶8邻域判断（未完成）
                    endPoints.append([i, j])
    return endPoints

def getBranch(imgArr, endPoint):
    """
    通过端点获取分支, 通过广度优先搜索
    args:
        imgArr : 二值化后的图像
        endPoint : 端点坐标
    """
    # print("endPoint:", endPoint.shape)
    endPoint = endPoint.tolist()
    # 遍历每个端点
    if not endPoint:
        return []
    branchs = []
    for point in endPoint:  # point = [i, j]
        # 通过广度优先搜索获取分支
        branch = []
        queue = deque()
        queue.append(point)  # 队列初始化

        while queue:
            point = queue.popleft()  # 获取队列首元素

            if point in branch:  # 判断是否已经遍历过
                continue

            branch.append(point)  # 出队的点添加到分支（保证点为有效点）

            # 获取八邻域
            neighbors = [[point[0]-1, point[1]], [point[0]+1, point[1]], [point[0], point[1]-1], [point[0], point[1]+1], 
                         [point[0]-1, point[1]-1], [point[0]-1, point[1]+1], [point[0]+1, point[1]-1], [point[0]+1, point[1]+1]]
            
            connect = 0  # 判断该点是几邻域联通
            for neighbor in neighbors:
                connect += imgArr[neighbor[0], neighbor[1]]
            if connect > 2:
                break  # 如果大于1，说明该点是交叉点，跳出while循环

            for neighbor in neighbors:
                if imgArr[neighbor[0], neighbor[1]] == 0:
                    continue

                else:
                    queue.append(neighbor)
        branchs.append((branch))
    return branchs

def removeBranch(img_bin, branch):
    """
    移除分支, 分支适当扩张
    args:
        img_bin : 二值化后的图像
        branch : 分支
    """
    # 分支扩张
    for point in branch:
        # 获取八邻域
        neighbors = [[point[0]-1, point[1]], [point[0]+1, point[1]], [point[0], point[1]-1], [point[0], point[1]+1], 
                     [point[0]-1, point[1]-1], [point[0]-1, point[1]+1], [point[0]+1, point[1]-1], [point[0]+1, point[1]+1]]
        for neighbor in neighbors:
            img_bin[neighbor[0], neighbor[1]] = 1
    return img_bin
def pts2shp(image_path, strVectorFile, pts):
    # 点矢量化
    ds, GeoTrans, _, _, _, imgArr = getTiff(image_path)
    toshp(ds, GeoTrans, pts, strVectorFile)
    print("finished")

def main():
    with open('/parcel_15cm/post_processing/processingEdge/seg_postprocess/contours.pkl', 'rb') as f:
        contours = pickle.load(f)  # 加载轮廓

    counters_processed = []  # 记录处理后的
    start_time = time.time()

    for i_counter in range(len(contours)):
        #------------------------------------------------
        # 处理点坐标
        #------------------------------------------------
        pts = contours[i_counter][:, 0, :]  # 获取点坐标[x, 1, y] -> [x, y]
        
        area = polygon_area(pts)*(0.15*0.15)
        if area < 14.18625:
            # print("面积小于阈值，跳过")
            continue

        x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
        x_min = x_min - 50
        y_min = y_min - 50

        # 归一化
        pts[:, 0] = pts[:, 0] - x_min
        pts[:, 1] = pts[:, 1] - y_min

        # 移到中间
        img = np.zeros((max(pts[:, 0].max(), pts[:, 1].max()) + 50, 
                        max(pts[:, 0].max() + 50, pts[:, 1].max()), 3), 
                        np.uint8
                        )
        
        #------------------------------------------------
        # 开始处理
        #------------------------------------------------

        cv2.polylines(img, [pts], True, (255, 0, 0), 1)  # 绘制线
        cv2.fillPoly(img, [pts], (255, 255, 255))  # 填充
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转为灰度图
        ret, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 二值化

        # 4. 膨胀-腐蚀：得到细线
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img_dilate = cv2.dilate(img_bin, kernel, iterations=1)
        img_erode = cv2.erode(img_bin, kernel, iterations=1)
        img_dilate_Minus_erode = img_dilate - img_erode

        # 5. 提取骨架
        img_dilate_Minus_erode[img_dilate_Minus_erode != 0] = 1
        img_dilate_Minus_erode_skeletonize = morphology.skeletonize(img_dilate_Minus_erode)

        # 6. 寻找端点、交点、及支线
        # 6.1 通过遍历邻域搜索端点
        endPoint= np.array(getEndPoint(img_dilate_Minus_erode_skeletonize))
        
        # 6.2 通过遍历邻域搜索交点
        branch= np.array(getBranch(img_dilate_Minus_erode_skeletonize, endPoint))

        #------------------------------------------------
        # 去掉指定的支线，延长指定的支线
        #------------------------------------------------
        #  计算每个branch的长度
        if branch.any():
            for i in range(branch.shape[0]):  # 遍历每个branch
                branch_start =  np.array(branch[i][-1])
                branch_end = np.array(branch[i][0])
                distance_each_branch = np.sqrt(((branch_start - branch_end)@(branch_start - branch_end).T))  # 每个branch的长度
                if distance_each_branch < 6:
                    # print("长度小于阈值，跳过")
                    img_bin = removeBranch(img_bin, branch[i])
                    
        # 对img_bin腐蚀并提取counters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img_erode = cv2.erode(img_bin, kernel, iterations=1)
        pts, _ = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        pts = np.array(pts)

        if len(pts.shape) == 4:
            pts = pts[0, ...]
            assert pts.shape[2] == 2, "pts.shape%d_%d" % pts.shape
            pts = pts[:, 0, :]

        #------------------------------------------------
        # 记录处理后的,并且还原
        #------------------------------------------------

            counters_processed.append(recovery_pts(pts, x_min, y_min))

    pts2shp(
        image_path = "/SAM_Various_Files/STL/dataset_1024X1024_s128_b4_0.989515/results.tif", 
        strVectorFile = "/SAM_Various_Files/STL/dataset_1024X1024_s128_b4_0.989515/remove_small_distance.shp", 
        pts = counters_processed,
        )
    end_time = time.time()
    print("time:", end_time - start_time)
    print("finished")

if __name__ == "__main__":
    main()