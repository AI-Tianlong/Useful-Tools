
"""
ATL_GDAL
---
包含了使用`GDAL`对遥感图像进行处理的一些工具

用法：
----
    >>> # 在开头复制这一句
    >>> from ATL_Tools import (read_img_to_array_with_info, # ✔读取影像为数组并返回信息
                               read_img_to_array,  # ✔读取影像为数组
                               save_ds_to_tif,     # ✔将GDAL dataset数据格式写入tif保存
                               save_array_to_tif,  # 将数组格式写入tif保存
                               read_img_get_geo,   # ✔计算影像角点的地理坐标或投影坐标
                               ds_get_img_geo,     # 读取dataset格式，计算影像角点的地理坐标或投影坐标
                               pix_to_geo,         # 计算影像某一像素点的地理坐标或投影坐标
                               geo_to_pix,         # 根据GDAL的仿射变换参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
                               Mosaic_all_imgs,    # ✔将指定路径文件夹下的tif影像全部镶嵌到一张影像上
                               Mosaic_2img_to_one, # 将两幅影像镶嵌至同一幅影像
                               raster_overlap,     # 两个栅格数据集取重叠区或求交集（仅测试方形影像）
                               crop_tif_with_json_zero, # ✔将带有坐标的图像按照json矢量进行裁切,无数据区域为0
                               crop_tif_with_json_nan,  # ✔将带有坐标的图像按照json矢量进行裁切,无数据区域为nan
                               Merge_multi_json,   # ✔将多个小的json合并为一个大的json,
                               resample_image      # ✔使用GDAL对图像进行重采样
                            )
                            
    ________________________________________________________________
    >>> # 示例1-读取影像为数组并返回信息
    >>> img = read_img_to_array_with_info(img_path) # 读取图片，并输出图片信息
    ________________________________________________________________
    >>> # 示例2-读取影像为数组
    >>> img = read_img_to_array(img_path) # 读取图片为数组
    ________________________________________________________________
    >>> # 示例3:

"""
#!/usr/bin/env python
# coding: utf-8

from osgeo import gdal, ogr
import os
import glob
import numpy as np
import math
from typing import Dict, List, Optional, Sequence
from ATL_path import mkdir_or_exist, find_data_list
from tqdm import tqdm 
import json

def read_img_to_array_with_info(filename: str, 
             convert_HWC: Optional[bool] = False) -> np.ndarray:   
    '''✔读取影像为数组并返回信息.

    Args:
        filename (str): 输入的影像路径
        convert_HWC (bool): 是否转换为 H*W*C 格式，默认为False 

    Returns: 
        影像的numpy数组格式，并显示影像的基本信息
    '''

    dataset = gdal.Open(filename) #打开文件    
    print('栅格矩阵的行数:', dataset.RasterYSize)
    print('栅格矩阵的列数:', dataset.RasterXSize)
    print('波段数:', dataset.RasterCount)
    print('数据类型:', dataset.GetRasterBand(1).DataType)
    print('仿射矩阵(左上角像素的大地坐标和像素分辨率)', dataset.GetGeoTransform())
    print('地图投影信息:', dataset.GetProjection())
    im_data = dataset.ReadAsArray()
    if convert_HWC:
        im_data = np.transpose(im_data, (1, 2, 0))

    del dataset 
    return im_data

def read_img_to_array(filename: str, 
             convert_HWC: Optional[bool] = False) -> np.ndarray:     # 读取影像为数组
    '''✔读取影像为数组.

    Args:
        filenam (str):输入的影像路径 
        convert_HWC (bool): 是否转换为 H*W*C 格式，默认为False 

    Returns: 
        影像的numpy数组格式
    '''

    dataset = gdal.Open(filename) #打开文件
    im_data = dataset.ReadAsArray()
    if convert_HWC:
        im_data = np.transpose(im_data, (1, 2, 0))
    
    del dataset 
    return im_data

def save_ds_to_tif(GDAL_dataset: gdal.Dataset,
           out_path: str,
           bands = None) -> None:
    """✔将GDAL dataset数据格式写入tif保存.

    Args:
        GDAL_dataset (gdal.Dataset)：输入的GDAL影像数据格式
        out_path (str)：输出的文件路径
        bands(List[str]): 输出的波段数，默认为None(输出所有波段

    Returns: 
        无 Return，输出影像文件至`out_path`
    """

    # 读取dataset信息
    im_array = GDAL_dataset.ReadAsArray()
    im_array = np.transpose(im_array, (1, 2, 0))
    print(im_array.shape)
    im_width = GDAL_dataset.RasterXSize
    im_height = GDAL_dataset.RasterYSize
    im_bands = GDAL_dataset.RasterCount   
    im_geotrans = GDAL_dataset.GetGeoTransform()  
    im_proj = GDAL_dataset.GetProjection()
    im_datatype = GDAL_dataset.GetRasterBand(1).DataType
    
    # 将dataset 写入 tif
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(out_path ,im_width, im_height,im_bands,im_datatype)
    ds.SetGeoTransform(im_geotrans) 
    ds.SetProjection(im_proj)
    
    for band_num in range(im_bands):
        band = ds.GetRasterBand(band_num + 1)
        band.WriteArray(im_array[:, :, band_num]) 
    # del ds
    
def save_array_to_tif(
        img_array: np.ndarray,
        out_path: str, 
        Transform = None, 
        Projection = None, 
        Band: int = 1, 
        Datatype: int = 6):
    """×将数组格式写入tif保存.

    Args:
        img_array (np.ndarry): 待保存的影像数组
        out_path (str): 输出的文件路径
        Transform：仿射矩阵六参数数组，默认为空,详细格式见GDAL。
        Projection ：投影，默认为空,详细格式见GDA
        Band (int): 波段数，默认为1
        Datatype (int): 保存数据格式（位深），默认为6，GDT_Float32

    Returns: 
        输出影像文件
    """

    h,w,c= img_array.shape
    print(img_array.shape)
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(out_path ,w, h, c, Datatype)
    if not Transform==None:
        ds.SetGeoTransform(Transform) 
    if not Projection==None:
        ds.SetProjection(Projection)  
    if not Band == None:
        Band = c

    for band_num in range(Band):
        band = ds.GetRasterBand(band_num + 1)
        band.WriteArray(img_array[:, :, band_num]) 
    # del ds
    
def read_img_get_geo(img_path: str):
    '''计算影像角点的地理坐标或投影坐标

    Args:
        img_path (str): 影像路径

    Returns: 
        min_x: x方向最小值
        max_y: y方向最大值
        max_x: x方向最大值
        min_y: y方向最小值
    '''

    ds=gdal.Open(img_path)
    geotrans=list(ds.GetGeoTransform())
    xsize=ds.RasterXSize 
    ysize=ds.RasterYSize
    min_x=geotrans[0]
    max_y=geotrans[3]
    max_x=geotrans[0]+xsize*geotrans[1]
    min_y=geotrans[3]+ysize*geotrans[5]
    ds=None
    
    return min_x, max_y, max_x, min_y

def ds_get_img_geo(GDAL_dataset: gdal.Dataset):
    '''读取dataset格式，计算影像角点的地理坐标或投影坐标

    Args:
        GDAL_dataset： GDAL dataset格式数据

    Returns: 
        min_x： x方向最小值
        max_y： y方向最大值
        max_x： x方向最大值
        min_y:  y方向最小值
    '''
    geotrans=list(GDAL_dataset.GetGeoTransform())
    xsize=GDAL_dataset.RasterXSize 
    ysize=GDAL_dataset.RasterYSize
    min_x=geotrans[0]
    max_y=geotrans[3]
    max_x=geotrans[0]+xsize*geotrans[1]
    min_y=geotrans[3]+ysize*geotrans[5]
    GDAL_dataset=None
    
    return min_x,max_y,max_x,min_y

def pix_to_geo(Xpixel: int, Ypixel: int, GeoTransform)->List[int]:
    '''计算影像某一像素点的地理坐标或投影坐标

    Args:
        Xpixel (int): 像素坐标x
        Ypixel (int): 像素坐标y
        GeoTransform：仿射变换参数

    Returns: 
        XGeo： 地理坐标或投影坐标X
        YGeo： 地理坐标或投影坐标Y
    '''

    XGeo = GeoTransform[0] + GeoTransform[1] * Xpixel + Ypixel * GeoTransform[2]
    YGeo = GeoTransform[3] + GeoTransform[4] * Xpixel + Ypixel * GeoTransform[5]
    return XGeo, YGeo

def geo_to_pix(dataset, x, y):
    '''根据GDAL的仿射变换参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）

    Args:
        dataset: GDAL地理数据
        x: 投影或地理坐标x
        y: 投影或地理坐标y 

    Returns: 
        影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    '''

    trans = dataset.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    
    return np.linalg.solve(a, b)
    
def Mosaic_all_imgs(img_file_path: str,
                    output_path: str,
                    nan_or_zero: str='zero') -> None:
    '''✔将指定路径文件夹下的tif影像全部镶嵌到一张影像上
        细节测试:
        ✔镶嵌根据矢量裁切后的图像，无数据区域为nan
        ✔镶嵌矩形的图像，无数据区域为nan

    注：将多个图合并之后，再进行裁切的话，nan就是白的，zero就是黑的
        如果单独裁切一个的话，不管nan还是zero都是白的
        如果将裁切后的进行合并的话，会把nan的部分也合并进去，需要单独处理

        需要进行优化
    
    Args:
        img_file_path (str)：tif 影像存放路径
        output_path (str): 输出镶嵌后 tif 的路径
        Nan_or_Zero (str): 'nan'或'zero'镶嵌后的无效数据nan或0,默认为0
                           'nan'更适合显示，'0更适合训练'
    Returns: 
        镶嵌合成的整体影像
    '''

    # os.chdir(img_file_path) # 切换到指定路径
    #如果存在同名影像则先删除
    if os.path.exists(output_path):
        print(f"存在{output_path}, 已覆盖")
        os.remove(output_path)

    all_files = find_data_list(img_file_path, suffix='.tif') # 寻找所有tif文件
    assert all_files!=None, 'No tif files found in the path'

    #获取待镶嵌栅格的最大最小的坐标值
    min_x,max_y,max_x,min_y=read_img_get_geo(all_files[0]) 
    for in_fn in all_files:
        minx, maxy, maxx, miny=read_img_get_geo(in_fn)
        min_x = min(min_x, minx)
        min_y = min(min_y, miny)
        max_x = max(max_x, maxx)
        max_y = max(max_y, maxy)
    # print("待镶嵌栅格的最大最小的坐标值")
    # print(f'min_x:{min_x}, min_y:{min_y}')
    # print(f'max_x:{max_x}, max_y:{max_y}')

    
    #计算镶嵌后影像的行列号
    in_ds=gdal.Open(all_files[0])
    geotrans=list(in_ds.GetGeoTransform())
    
    # 这一行代码获取了数据集的地理转换信息，包括地理坐标系的变换参数。
    # in_ds.GetGeoTransform() 返回一个包含六个浮点数的元组，
    # 分别表示左上角的X坐标、水平像素分辨率、X方向的旋转（通常为0）、
    # 左上角的Y坐标、Y方向的旋转（通常为0）、垂直像素分辨率。
    # 这一行代码将获取的元组转换为列表形式，并赋值给 geotrans。

    width_geo_resolution=geotrans[1]
    heigh_geo_resolution=geotrans[5]
    # print(f'width_geo_resolution:{width_geo_resolution}, heigh_geo_resolution:{heigh_geo_resolution}')

    columns=math.ceil((max_x-min_x)/width_geo_resolution) 
    rows=math.ceil((max_y-min_y)/(-heigh_geo_resolution))
    bands = in_ds.RasterCount
    print(f'新合并图像的尺寸: {rows, columns, bands}')

    in_band_DataType = in_ds.GetRasterBand(1).DataType

    driver=gdal.GetDriverByName('GTiff')
    out_ds=driver.Create(output_path, columns, rows, bands, in_band_DataType)
    out_ds.SetProjection(in_ds.GetProjection())
    geotrans[0] = min_x
    geotrans[3] = max_y
    out_ds.SetGeoTransform(geotrans)


    #定义仿射逆变换
    inv_geotrans=gdal.InvGeoTransform(geotrans)

    #开始逐渐写入
    for in_fn in tqdm(all_files, desc='正在镶嵌图像ing...'):
        print('正在镶嵌:', os.path.abspath(in_fn))
        in_ds = gdal.Open(in_fn)
        in_gt = in_ds.GetGeoTransform()
        #仿射逆变换
        offset = gdal.ApplyGeoTransform(inv_geotrans, in_gt[0], in_gt[3])
        x, y = map(int, offset)
        # print(f'逆变换后的像素：x:{x}, y:{y}')
        # 该函数返回一个转换器对象 trans，可以使用这个对象执行从输入数据集到输出数据集的坐标转换
        trans = gdal.Transformer(in_ds, out_ds, [])       #in_ds是源栅格，out_ds是目标栅格
        success, xyz = trans.TransformPoint(False, 0, 0)  #计算in_ds中左上角像元对应out_ds中的行列号
        x, y, z = map(int, xyz)
        # print(f'小图(0, 0)变换到大图的像素：(x:{x}, y:{y}, z:{z})')

        for band_num in range(bands):
            # 小图的单通道，(h,w),无数据全是nan
            in_ds_array = in_ds.GetRasterBand(band_num + 1).ReadAsArray() #(h,w)
            # 无效数据的地方全是nan,这也符合下载下来的图，空缺的地方是nan、
            # 把nan的地方，用0替代
            in_ds_array = np.nan_to_num(in_ds_array, nan=0.)

            # 大图的单通道，(h,w)，没数据的地方全是0
            big_out_band = out_ds.GetRasterBand(band_num + 1)
            # 大图中，小图区域的数据
            Tiny_in_BigOut_data = big_out_band.ReadAsArray(x, y, in_ds_array.shape[1], in_ds_array.shape[0])
            Tiny_in_BigOut_data = np.nan_to_num(Tiny_in_BigOut_data, nan=0.)
            # 最后要写入大图的数据：如果是根据矢量裁切完的应该不会有重合，直接相加就行
            # 但是如果是矩形大图的话，有重叠的话，则需要舍弃小图的重叠部分。
            # 第一步：找到大图中有数据的区域：即大图不为零的地方
            # 第二步：利用大图中不为零的地方，把小图中的值设置为0 
            # 第三部：把两个图相加，得到最后的结果
            zero_mask_in_tiny_of_big = Tiny_in_BigOut_data!=0.
            in_ds_array[zero_mask_in_tiny_of_big] = 0.
            in_ds_array = Tiny_in_BigOut_data + in_ds_array

            # 写入大图
            big_out_band.WriteArray(in_ds_array, x, y)
                        
            # 最后把所有为0.的地方都变成nan
            if nan_or_zero == 'nan':
                big_out_band_nan = big_out_band.ReadAsArray()
                big_out_band_nan[big_out_band_nan==0] = np.nan
                big_out_band.WriteArray(big_out_band_nan)

    del in_ds, out_ds   
    print(f'镶嵌图像已完成，输出至 {output_path}')
     

def Mosaic_2img_to_one(ds1 , ds2, path):
    '''将两幅影像镶嵌至同一幅影像

    Args:
        ds1：镶嵌数据集1
        ds2：镶嵌数据集1

    Returns: 
        镶嵌合成的整体影像
    '''
    band1 = ds1.GetRasterBand(1)
    rows1 = ds1.RasterYSize
    cols1 = ds1.RasterXSize
    
    band2 = ds2.GetRasterBand(1)
    rows2 = ds2.RasterYSize
    cols2 = ds2.RasterXSize
    
    (minX1,maxY1,maxX1,minY1) = ds_get_img_geo(ds1)
    (minX2,maxY2,maxX2,minY2) = ds_get_img_geo(ds2)


    transform1 = ds1.GetGeoTransform()
    pixelWidth1 = transform1[1]
    pixelHeight1 = transform1[5] #是负值（important）
    
    transform2 = ds2.GetGeoTransform()
    pixelWidth2 = transform1[1]
    pixelHeight2 = transform1[5] 
    
    # 获取输出图像坐标
    minX = min(minX1, minX2)
    maxX = max(maxX1, maxX2)
    minY = min(minY1, minY2)
    maxY = max(maxY1, maxY2)

    #获取输出图像的行与列
    cols = int((maxX - minX) / pixelWidth1)
    rows = int((maxY - minY) / abs(pixelHeight1))

    # 计算图1左上角的偏移值（在输出图像中）
    xOffset1 = int((minX1 - minX) / pixelWidth1)
    yOffset1 = int((maxY1 - maxY) / pixelHeight1)

    # 计算图2左上角的偏移值（在输出图像中）
    xOffset2 = int((minX2 - minX) / pixelWidth1)
    yOffset2 = int((maxY2 - maxY) / pixelHeight1)

    # 创建一个输出图像
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create( path, cols, rows, 1, band1.DataType)#1是bands，默认
    out_band = out_ds.GetRasterBand(1)

    # 读图1的数据并将其写到输出图像中
    data1 = band1.ReadAsArray(0, 0, cols1, rows1)
    out_band.WriteArray(data1, xOffset1, yOffset1)

    #读图2的数据并将其写到输出图像中
    data2 = band2.ReadAsArray(0, 0, cols2, rows2)
    out_band.WriteArray(data2, xOffset2, yOffset2)
    ''' 写图像步骤'''
    
    #第二个参数是1的话：整幅图像重度，不需要统计
    # 设置输出图像的几何信息和投影信息
    geotransform = [minX, pixelWidth1, 0, maxY, 0, pixelHeight1]
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(ds1.GetProjection())
    
    del ds1,ds2,out_band,out_ds,driver
   
    return 0

def raster_overlap(ds1, ds2, nodata1=None, nodata2=None):
    '''两个栅格数据集取重叠区或求交集（仅测试方形影像）

    Args:
        ds1 (GDAL dataset) - GDAL dataset of an image
        ds2 (GDAL dataset) - GDAL dataset of an image
        nodata1 (number) - nodata value of image 1
        nodata2 (number) - nodata value of image 2
        
    Returns: 
        ds1c (GDAL dataset), ds2c (GDAL dataset): 011
    '''

##Setting nodata
    nodata = 0
    ###Check if images NoData is set
    if nodata2 is not None:
        nodata = nodata2
        ds2.GetRasterBand(1).SetNoDataValue(nodata)
    else:
        if ds2.GetRasterBand(1).GetNoDataValue() is None:
            ds2.GetRasterBand(1).SetNoDataValue(nodata)

    if nodata1 is not None:
        nodata = nodata1
        ds1.GetRasterBand(1).SetNoDataValue(nodata1)
    else:
        if ds1.GetRasterBand(1).GetNoDataValue() is None:
            ds1.GetRasterBand(1).SetNoDataValue(nodata)

    ### Get extent from ds1
    projection = ds1.GetProjection()
    geoTransform = ds1.GetGeoTransform()

    ###Get minx and max y
    
    [minx, maxy, maxx, miny] = ds_get_img_geo(ds1)
    [minx_2, maxy_2, maxx_2, miny_2] = ds_get_img_geo(ds2)
    
    min_x = sorted([maxx,minx_2,minx,maxx_2])[1]    # 对边界值排序，第二三个为重叠区边界
    max_y = sorted([maxy,miny_2,miny,maxy_2])[2]
    max_x = sorted([maxx,minx_2,minx,maxx_2])[2]
    min_y = sorted([maxy,miny_2,miny,maxy_2])[1]
    
    ###Warp to same spatial resolution
    gdaloptions = {'format': 'MEM', 'xRes': geoTransform[1], 'yRes': 
    geoTransform[5], 'dstSRS': projection}
    ds2w = gdal.Warp('', ds2, **gdaloptions)
    ds2 = None

    ###Translate to same projection
    ds2c = gdal.Translate('', ds2w, format='MEM', projWin=[min_x, max_y, max_x, min_y], 
    outputSRS=projection)
    ds2w = None
    ds1c = gdal.Translate('', ds1, format='MEM', projWin=[min_x, max_y, max_x, min_y], 
    outputSRS=projection)
    ds1 = None

    return ds1c,ds2c

def crop_tif_with_json_zero(img_path: str,
                       output_path: str,
                       geojson_path: str):
    '''✔将带有坐标的图像按照json矢量进行裁切,矢量外无数据区域为0

    Args:
        img_path (str): 输入图像的路径
        output_path (str): 输出图像的路径
        geojson_path (str): Geojson文件的路径
        
    Returns: 
        保存裁切后的图像至本地
    '''
    if os.path.exists(output_path):
        print(f"存在{output_path}, 已覆盖")
        os.remove(output_path)

    # 打开栅格文件
    raster_ds = gdal.Open(img_path)
    assert raster_ds!=None, f'打开 {raster_ds} 失败'

    # 打开GeoJSON文件
    geojson_ds = ogr.Open(geojson_path)
    geojson_layer = geojson_ds.GetLayer()

    # 获取GeoJSON文件的范围
    xmin, xmax, ymin, ymax = geojson_layer.GetExtent()
    # 设置裁剪范围
    warp_options = gdal.WarpOptions(cutlineDSName=geojson_path,
                                    cutlineWhere=None,
                                    cropToCutline=None,
                                    outputBounds=(xmin, ymin, xmax, ymax),
                                    dstSRS='EPSG:4326')  # 设置输出投影，这里使用EPSG:4326，即WGS84经纬度坐标系

    # 执行裁剪
    gdal.Warp(output_path, raster_ds, options=warp_options)

    # 关闭数据源
    raster_ds = None
    geojson_ds = None
    print(f'根据矢量裁切{img_path}完成！无数据区域为0')

def crop_tif_with_json_nan(img_path: str,
                           output_path: str,
                           geojson_path: str) -> None:
    '''✔将带有坐标的图像按照json矢量进行裁切
    使无数据区域的值为nan,优先使用这个, 矢量外无数据区域为nan

    Args:
        img_path (str): 输入图像的路径
        output_path (str): 输出图像的路径
        geojson_path (str): Geojson文件的路径
        
    Returns: 
        保存裁切后的图像至本地
    '''

    if os.path.exists(output_path):
        print(f"存在{output_path}, 已覆盖")
        os.remove(output_path)
    # 打开栅格文件
    raster_ds = gdal.Open(img_path)
    assert raster_ds!=None, f'打开 {raster_ds} 失败'

    # 打开GeoJSON文件
    geojson_ds = ogr.Open(geojson_path)
    geojson_layer = geojson_ds.GetLayer()

    # 获取GeoJSON文件的范围
    xmin, xmax, ymin, ymax = geojson_layer.GetExtent()
    # 设置裁剪范围
    warp_options = gdal.WarpOptions(cutlineDSName=geojson_path,
                                    cutlineWhere=None,
                                    cropToCutline=None,
                                    outputBounds=(xmin, ymin, xmax, ymax),
                                    dstSRS='EPSG:4326',# 设置输出投影，这里使用EPSG:4326，即WGS84经纬度坐标系
                                    creationOptions=['COMPRESS=DEFLATE', 'TILED=YES', 'BIGTIFF=YES', 'NUM_THREADS=ALL_CPUS', 'ALPHA=YES'],
                                    dstNodata=float('nan')  # 设置裁剪后的无数据值为 NaN
                                    )  

    # 执行裁剪
    gdal.Warp(output_path, raster_ds, options=warp_options)

    # 关闭数据源
    raster_ds = None
    geojson_ds = None
    print(f'根据矢量裁切{img_path}完成！无数据区域为 NaN')

def Merge_multi_json(input_json_file: str,
                     output_json: str) -> None:
    """✔将多个小的json合并为一个大的json

    Args:
        input_json_file (str): 要合并的json文件的路径
        output_json (str): 合并后输出的json文件名
    """
    
    # 读取json文件
    def read_json(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    
    json_file = find_data_list(input_json_file, '.json')
    assert json_file != None, f"{json_file} 下 未找到 .json 文件"

    new_json_features = []
    for tiny_json_name in tqdm(json_file):
        tiny_json = read_json(tiny_json_name)
        tiny_features =  tiny_json['features'][0]
        new_json_features.append(tiny_features)
    # print(new_json_features)
        # print(tiny_features)

    new_json_content = {
        "type": "FeatureCollection",
        "features": new_json_features
    }

    with open(output_json, 'w') as f:
        json.dump(new_json_content, f)
    print(f"合并json {output_json} 文件完成！")

def resample_image(input_path: str, 
                   output_path: str, 
                   scale_factor: str):
    """✔使用GDAL对图像进行重采样
    
    Args:
        input_path (str): 输入图像路径
        output_path (str): 输出图像路径
        scale_factor (float): 缩放因子

    Returns:
        输出重采样后的图像
    """

    # 打开输入图像
    input_ds = gdal.Open(input_path)

    # 获取输入图像的宽度和高度
    cols = input_ds.RasterXSize
    rows = input_ds.RasterYSize

    # 计算输出图像的新宽度和新高度
    new_cols = int(cols * scale_factor)
    new_rows = int(rows * scale_factor)

    # 使用gdal.Warp函数进行重采样
    gdal.Warp(output_path, input_ds, format='GTiff', width=new_cols, height=new_rows)

    # 关闭数据集
    input_ds = None

    





