'''
Description: 这个版本只去除因边界叠加造成的裂口
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
import pickle
import time
from collections import deque

class Point():
    """
    点类
    """
    x = 0.0
    y = 0.0
    index = 0  # 点在线上的索引

    def __init__(self, x, y, index):
        self.x = x
        self.y = y
        self.index = index


class Douglas():
    """ 道格拉斯算法 """

    def __init__(self, distThre=4):
        """
        distThre: 这个是像素之间的容差
        """
        self.points = []
        self.D = distThre  # 容差, 坐标之间的距离（坐标是像素中心点，因此此处可以看作是像素个数之间的距离）
        self.pointsout = []

    def readPoint(self, contour):
        """生成点要素"""
        for i in range(len(contour)):
            self.points.append(Point(x=contour[i, 0], y=contour[i, 1], index=i))  # 取每个坐标的(x, y), 并加上索引
        self.pointsout = self.points

    def compress(self, p1, p2):
        """
        具体的抽稀算法
        p1 = (Xp1, Yp1) -> (p1.x, p1.y)
        p2 = (Xp2, Yp2) -> (p2.x, p2.y)
        """
        swichvalue = False
        # 一般式直线方程系数 A*x+B*y+C=0, 利用两点求解方程系数
        A = (p1.y - p2.y)
        B = (p2.x - p1.x)
        C = (p1.x * p2.y - p2.x * p1.y)

        m = self.points.index(p1)  # 列表第一个点
        n = self.points.index(p2)  # 列表最后一个点
        distance = []
        middle = None
        if n == m + 1:  # 也就是只有两个点
            return

        # 计算中间点与直线的距离
        for i in range(m + 1, n):
            d = abs(A * self.points[i].x + B * self.points[i].y + C) / math.sqrt(math.pow(A, 2) + math.pow(B, 2))  # 点到直线的距离
            distance.append(d)
        dmax = max(distance)
        if dmax > self.D:
            swichvalue = True
        else:
            swichvalue = False

        if not swichvalue:
            t = 0
            for i in range(m + 1, n):
                del self.points[i - t]
                t = t + 1
        else:
            for i in range(m + 1, n):
                if distance[i - m - 1] == dmax:  # abs(A*self.points[i].x+B*self.points[i].y+C)/math.sqrt(math.pow(A,2)+math.pow(B,2))
                    middle = self.points[i]
                    break
            self.compress(p1, middle)
            self.compress(middle, p2)


def DouglasPeuker(contours, Thre):  # 道格拉斯抽稀
    """
    contours: [轮廓个数, 点个数, 1 , XY]
    Thre: 抽稀的容差
    """
    print("DouglasPeuker ...")
    sparsePoints = []
    d = Douglas(distThre=Thre)
    for _, contour in enumerate(contours):
        d.points = []
        contour = np.squeeze(contour)
        # contour = shapelysimplify(contour)
        d.readPoint(contour)  # [[x, y, index], ...]
        d.compress(d.points[0], d.points[len(d.points) - 1])  # 当前点（x, y, index）和上一个点（x, y, index）
        pointssize = len(d.points)
        plist = np.zeros((pointssize, 2), dtype=np.int32)
        for i in range(pointssize):
            plist[i, 0] = d.points[i].x
            plist[i, 1] = d.points[i].y
        sparsePoints.append(plist)
    return sparsePoints

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

def main(pixel_res=0.2):
    ds, GeoTrans, _, _, _, imgArr = getTiff(imgPath="/SAM_Various_Files/STL/dataset_1024X1024_s128_b4_0.989515/results.tif")
    imgArr[imgArr < 0.005758265] = 0
    imgArr[imgArr != 0] = 1
    # 计算轮廓,以像素坐标点表示
    contours, _ = cv2.findContours(imgArr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # < contours：返回的轮廓像素坐标点; hierarchy：每条轮廓对应的属性 >

    counters_processed = []  # 记录处理后的
    start_time = time.time()

    for i_counter in tqdm(range(len(contours))):
        #------------------------------------------------
        # 处理点坐标
        #------------------------------------------------
        i_counter = i_counter.astype('float64')
        pts = contours[i_counter][:, 0, :]  # 获取点坐标[x, 1, y] -> [x, y]
        
        area = polygon_area(pts)*(pixel_res*pixel_res)
        if area < 15:
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
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        img_dilate = cv2.dilate(img_bin, kernel, iterations=3)
        img_erode = cv2.erode(img_dilate, kernel, iterations=3)

        pts, _ = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 获取轮廓 (1, 849, 1, 2)
        
        pts = list(pts)
        for i in range(len(pts)):   
            pt = np.array(pts[i])
            pt = pt[:, 0, :]
            pt = recovery_pts(pt, x_min, y_min)
            pt = pt[:, np.newaxis, :].tolist()
            counters_processed.append(pt)

    counters_processed = DouglasPeuker(counters_processed, Thre=6)

        #------------------------------------------------
        # 点集后处理
        #------------------------------------------------

        #------------------------------------------------
        # 还原成矢量
        #------------------------------------------------
    pts2shp(
        image_path = "/SAM_Various_Files/STL/dataset_1024X1024_s128_b4_0.989515/results.tif", 
        strVectorFile = "/SAM_Various_Files/STL/dataset_1024X1024_s128_b4_0.989515/remove_small_distance_DouglasPeuker.shp", 
        pts = counters_processed,
        )
    end_time = time.time()
    print("time:", end_time - start_time)
    print("finished")

if __name__ == "__main__":
    main()