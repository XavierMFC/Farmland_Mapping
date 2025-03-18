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


def getTiff(imgPath):
    ds = gdal.Open(imgPath)
    GeoTrans = ds.GetGeoTransform()  # 仿射矩阵
    ref_width = ds.RasterXSize  # 栅格矩阵的列数
    ref_height = ds.RasterYSize  # 栅格矩阵的行数
    ref_Proj = ds.GetProjection()  # 投影信息
    imgArr = ds.ReadAsArray(0, 0, ref_width, ref_height)  # 将数据写成数组，对应栅格矩阵
    print(np.unique(imgArr))
    return ds, GeoTrans, ref_width, ref_height, ref_Proj, imgArr


def arr2tiff(ref, save_path, arr):
    # plt.imsave(save_path, arr)  # 保存为jpg格式
    _, GeoTrans, ref_width, ref_height, ref_Proj, _ = getTiff(ref)
    # 保存为TIF格式
    driver = gdal.GetDriverByName("GTiff")
    assert (ref_width, ref_height) == (arr.shape[1], arr.shape[2]), print('栅格尺寸与参考影像不匹配!')
    datasetnew = driver.Create(save_path, ref_width, ref_height, 1, gdal.GDT_CInt16)
    datasetnew.SetGeoTransform(GeoTrans)
    datasetnew.SetProjection(ref_Proj)
    band = datasetnew.GetRasterBand(1)
    band.WriteArray(arr)
    datasetnew.FlushCache()  # Write to disk.必须清除缓存
    print('finished!')


def morph(img, value, areaThreshold, connectivity, kernel):
    """
    img: 输入的原始影像
    value: 指定处理的类别
    areaThreshold: 去除的面积阈值
    connectivity: 连通区域设定
    kernel: 形态学运算的核大小
    """
    # 1. 去除面积小于指定值的区域
    contours = []
    img = img == value
    img = morphology.remove_small_objects(img, min_size=areaThreshold, connectivity=connectivity)
    img = np.array(img * 1, dtype=np.uint8)

    # 2. 形态学运算，去除小点
    kernel = np.ones(kernel, np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)  # Close little holes  闭运算去除空洞
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)  # Open to erode small patches  开运算腐蚀小点
    
    img = img.astype('uint8')

    # 3. 计算斑块数量
    if connectivity == 1:
        numPatch = generate_binary_structure(2, 1)  # 四连通
    if connectivity == 2:
        numPatch = generate_binary_structure(2, 2)  # 八连通

    # 4. 计算轮廓,以像素坐标点表示
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # < contours：返回的轮廓像素坐标点; hierarchy：每条轮廓对应的属性 >

    # 5. 再次去除小区域
    # 可改进：此处去除小面积可以先统计然后根据统计值设置阈值去除
    contoursNew = []
    for i in range(len(contours)):
        contour = contours[i]
        area = cv2.contourArea(contour)
        print(area)
        if area > areaThreshold:  # 面积大于n才保存
            contoursNew.append(contour)

    return contoursNew


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


def cal_convexHull(points_input):
    """
    计算凸包
    Args:
        points_input: shape=list[point_nums, 2]
    Returns: shape=list[point_nums, 2]
    """
    # 计算凸包
    res = []
    for i in range(len(points_input)):
        hull = cv2.convexHull(points_input[i].reshape(-1, 1, 2))
        res.append(hull[:, 0, :])  # [points_num, x_y]
    return res


def cal_defectpoints(points_input):
    """
    计算凸包角点的索引
    Args:
        points_input: shape=list[point_nums, 2]
    Returns: shape=list[point_nums, 2]
    """

    res = []
    for i in range(len(points_input)):
        tmp = []
        hull = cv2.convexHull(points_input[i].reshape(-1, 1, 2), returnPoints=False)
        for point in hull:
            tmp.append(points_input[i][point].tolist()[0])
        res.append(np.array(tmp))
    return res


def select_diff_dp_convex(points1, points2, thre=9 / 10):
    """
    Args:
        points1: 凸包点 list[point_nums, 2]
        points2: 原始值 list[point_nums, 2]
        thre: 原始面积/凸包面积: 凸包面积一定>=原始面积 9/10为面积差不超10%
    Returns:

    """
    # 根据凸包与原点集的面积比进行筛选
    contours_convex = points1  # 凸包点
    contours_dp = points2  # 原始值
    contours_convex_selected_1 = []  # 满足要求的凸包
    contours_convex_selected_2 = []  # 不满足要求的凸包
    for i in range(len(contours_convex)):
        contour_dp_area = cv2.contourArea(contours_dp[i])
        contour_convex_area = cv2.contourArea(contours_convex[i])
        if contour_dp_area / contour_convex_area < thre:
            contours_convex_selected_2.append((i, contours_convex[i]))
            continue
        contours_convex_selected_1.append((i, contours_convex[i]))
    return contours_convex_selected_1, contours_convex_selected_2


def raster2shp(image_path, strVectorFile, kernel, areaThreshold=32):
    """
    image_path: 待矢量化的影像
    strVectorFile: 矢量化文件保存路径
    kernel: 形态学运算的核大小
    value: 指定处理的类别
    areaThreshold: 去除的面积阈值
    """
    # 1. 获取影像的数据、投影属性
    ds, GeoTrans, _, _, _, imgArr = getTiff(image_path)
    # 2. 形态学处理，返回以像素坐标点表示的轮廓 [[[[p1, p2]], [[p1, p2]]], ...]
    contours_morph = morph(imgArr, value=1, areaThreshold=areaThreshold, connectivity=1, kernel=kernel)  # 目标像素值,去除小区域面积阈值,连通性(1->4; 2->8)
    # 3. 道格拉斯算法简化:后处理的原始值
    # contours_dp = [points_num, x, y] = [np.array[[x1, y1], ...], np.array([[x1, y1], ...])]
    contours_dp = DouglasPeuker(contours_morph, Thre=2)

    # # 使用凸包检测将点分成两类
    # contours_convex = cal_convexHull(contours_dp)
    # # 将点集分为两部分 1是满足面积的，2是不满足
    # contours_convex_selected_1, contours_convex_selected_2 = select_diff_dp_convex(contours_convex, contours_dp)

    # . 矢量化处理得到的点
    _, _ = toshp(ds, GeoTrans, contours_dp, strVectorFile)
    return


if __name__ == "__main__":
    # 单图矢量化
    # 
    imgpath = "/SAM_Various_Files/STL/dataset_1024X1024_s128_b4_0.989515/results.tif"
    raster2shp(
        imgpath, 
        "/SAM_Various_Files/STL/dataset_1024X1024_s128_b4_0.989515/results_EliminatePolygo2R_k_9_9.shp", 
        kernel=(9, 9), 
        areaThreshold=50
        )
    print('finished !')
