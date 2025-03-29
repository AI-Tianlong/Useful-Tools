from osgeo import gdal
import numpy as np
import os

# 输入和输出文件路径
input_file = '../S2_GF2_PMS1__L1A0000564539-MSS1_10bands_merged.tif'
output_file = '../S2_GF2_PMS1__L1A0000564539-MSS1_TCI.tif'

def create_tci_image(input_path, output_path):
    """
    将多波段Uint16的卫星图像转换为0-255的TCI彩色图像
    
    参数:
    input_path - 输入的多波段图像路径
    output_path - 输出的TCI图像路径
    """
    # 打开源文件
    src_ds = gdal.Open(input_path)
    if src_ds is None:
        print(f"错误: 无法打开文件 {input_path}")
        return
    
    # 获取图像信息
    width = src_ds.RasterXSize
    height = src_ds.RasterYSize
    geo_transform = src_ds.GetGeoTransform()
    projection = src_ds.GetProjection()
    
    # 波段索引 (注意: GDAL波段索引从1开始)
    # Sentinel-2的B04(红)、B03(绿)、B02(蓝)波段通常对应于R-G-B
    # 在我们的merged文件中，波段索引应该是: 
    red_band_idx = 3    # B04
    green_band_idx = 2  # B03
    blue_band_idx = 1   # B02
    
    # 读取RGB波段数据
    red = src_ds.GetRasterBand(red_band_idx).ReadAsArray().astype(np.float32)
    green = src_ds.GetRasterBand(green_band_idx).ReadAsArray().astype(np.float32)
    blue = src_ds.GetRasterBand(blue_band_idx).ReadAsArray().astype(np.float32)
    
    # 打印原始值范围，帮助确定缩放参数
    print(f"红波段范围: {np.min(red)} - {np.max(red)}")
    print(f"绿波段范围: {np.min(green)} - {np.max(green)}")
    print(f"蓝波段范围: {np.min(blue)} - {np.max(blue)}")
    
    # 确定量化参数
    # 如果已经是float32，可能数值范围在0-1之间，需要乘以10000
    # 如果是原始的Uint16且未缩放，最大值可能在10000-65535之间
    if np.max(red) <= 1.0 and np.max(green) <= 1.0 and np.max(blue) <= 1.0:
        # 数据已经被缩放到0-1范围
        scale_factor = 255
        # 执行线性拉伸以增强对比度
        percentile_low, percentile_high = 2, 98
        
        # 通过百分位数计算拉伸范围
        min_r = np.percentile(red, percentile_low)
        max_r = np.percentile(red, percentile_high)
        min_g = np.percentile(green, percentile_low)
        max_g = np.percentile(green, percentile_high)
        min_b = np.percentile(blue, percentile_low)
        max_b = np.percentile(blue, percentile_high)
        
        # 线性拉伸并缩放到0-255
        red = np.clip((red - min_r) / (max_r - min_r) * scale_factor, 0, 255)
        green = np.clip((green - min_g) / (max_g - min_g) * scale_factor, 0, 255)
        blue = np.clip((blue - min_b) / (max_b - min_b) * scale_factor, 0, 255)
    else:
        # 假设是Uint16原始数据，典型的Sentinel-2 L1C反射率值在0-10000范围
        # 确定一个适当的拉伸范围
        max_value = 10000.0 if np.max([np.max(red), np.max(green), np.max(blue)]) < 20000 else 65535.0
        
        # 执行线性拉伸以增强对比度
        percentile_low, percentile_high = 2, 98
        
        # 通过百分位数计算拉伸范围
        min_r = np.percentile(red, percentile_low)
        max_r = np.percentile(red, percentile_high)
        min_g = np.percentile(green, percentile_low)
        max_g = np.percentile(green, percentile_high)
        min_b = np.percentile(blue, percentile_low)
        max_b = np.percentile(blue, percentile_high)
        
        # 线性拉伸并缩放到0-255
        red = np.clip((red - min_r) / (max_r - min_r) * 255, 0, 255)
        green = np.clip((green - min_g) / (max_g - min_g) * 255, 0, 255)
        blue = np.clip((blue - min_b) / (max_b - min_b) * 255, 0, 255)
    
    # 转换为uint8类型
    red = red.astype(np.uint8)
    green = green.astype(np.uint8)
    blue = blue.astype(np.uint8)
    
    # 创建目标文件
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(output_path, width, height, 3, gdal.GDT_Byte, 
                          options=["COMPRESS=LZW", "TILED=YES"])
    
    if dst_ds is None:
        print(f"错误: 无法创建输出文件 {output_path}")
        return
    
    # 设置空间参考信息
    dst_ds.SetGeoTransform(geo_transform)
    dst_ds.SetProjection(projection)
    
    # 写入RGB波段
    dst_ds.GetRasterBand(1).WriteArray(red)
    dst_ds.GetRasterBand(2).WriteArray(green)
    dst_ds.GetRasterBand(3).WriteArray(blue)
    
    # 设置波段颜色解释
    dst_ds.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
    dst_ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
    dst_ds.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
    
    # 构建金字塔
    dst_ds.BuildOverviews("AVERAGE", [2, 4, 8, 16, 32])
    
    # 关闭数据集
    src_ds = None
    dst_ds = None
    
    print(f"TCI图像生成完成! 输出文件: {output_path}")

if __name__ == "__main__":
    print("开始生成TCI彩色图像...")
    create_tci_image(input_file, output_file)