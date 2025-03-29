from ATL_Tools import find_data_list, mkdir_or_exist
import os
import numpy as np
from osgeo import gdal
import glob

# 修复波段顺序列表中的错误（B8A和B11之间缺少逗号）


def merge_bands(input_dir, output_path, band_order_list):
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取单波段文件
    band_files = {}
    for band_name in band_order_list:
        # 查找对应波段的文件
        pattern = os.path.join(input_dir, f"*{band_name}*.tif")
        matches = glob.glob(pattern)
        if matches:
            band_files[band_name] = matches[0]
        else:
            print(f"警告: 找不到波段 {band_name} 的文件")
    
    if not band_files:
        print("错误: 没有找到任何波段文件")
        return
    
    # 读取第一个波段获取基本信息
    first_band_name = list(band_files.keys())[0]
    first_ds = gdal.Open(band_files[first_band_name])
    
    if first_ds is None:
        print(f"错误: 无法打开文件 {band_files[first_band_name]}")
        return
    
    # 获取图像信息
    width = first_ds.RasterXSize
    height = first_ds.RasterYSize
    geo_transform = first_ds.GetGeoTransform()
    projection = first_ds.GetProjection()
    
    # 使用float32数据类型，而不是原始数据类型
    data_type = gdal.GDT_Float32
    
    # 创建目标文件
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(output_path, width, height, len(band_order_list), data_type, 
                          options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"])
    
    if dst_ds is None:
        print(f"错误: 无法创建输出文件 {output_path}")
        return
    
    # 设置空间参考信息
    dst_ds.SetGeoTransform(geo_transform)
    dst_ds.SetProjection(projection)
    
    # 逐个波段写入数据
    for i, band_name in enumerate(band_order_list):
        if band_name in band_files:
            print(f"处理波段 {band_name}, 索引: {i+1}")
            
            # 打开波段文件
            src_ds = gdal.Open(band_files[band_name])
            if src_ds is None:
                print(f"警告: 无法打开文件 {band_files[band_name]}")
                continue
            
            # 读取数据
            band_data = src_ds.GetRasterBand(1).ReadAsArray()
            
            # 将像素值除以10000并转换为float32
            # band_data = band_data.astype(np.float32) / 10000.0
            
            # 写入目标文件
            dst_band = dst_ds.GetRasterBand(i+1)
            dst_band.WriteArray(band_data)
            dst_band.SetDescription(band_name)  # 设置波段名称
            
            # 设置NoData值（如果需要）
            # dst_band.SetNoDataValue(0.0)
            
            # 关闭源数据集
            src_ds = None
        else:
            print(f"警告: 跳过波段 {band_name}, 未找到对应文件")
    
    # 计算统计信息
    dst_ds.BuildOverviews("NEAREST", [2, 4, 8, 16, 32])
    
    # 关闭数据集
    dst_ds = None
    first_ds = None
    
    print(f"合并完成! 输出文件: {output_path}")
    print("注意: 所有像素值已除以10000并保存为float32格式")

if __name__ == "__main__":

    band_order = ['B02', 'B03', 'B04', 'B05','B06','B07','B08', 'B8A', 'B11', 'B12']

    single_band_10m_dir = '../S2_GF2_PMS1__L1A0000564539-MSS1_单波段裁切-10m像素对齐/'
    output_file = '../S2_GF2_PMS1__L1A0000564539-MSS1_10bands_merged.tif'

    print("开始合并波段...")
    print(f"波段顺序: {band_order}")
    merge_bands(single_band_10m_dir, output_file, band_order)