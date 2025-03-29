from osgeo import gdal
import numpy as np
import os

# 输入和输出文件路径
input_file = '../S2_GF2_PMS1__L1A0000564539-MSS1_10bands_merged.tif'
output_file = '../S2_GF2_PMS1__L1A0000564539-MSS1_TCI-S2官网.tif'

def create_tci_image(input_path, output_path):
    """
    将多波段Uint16的卫星图像转换为0-255的TCI彩色图像，
    完全按照Sentinel-2官方的JavaScript脚本实现
    
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
    red_band_idx = 3    # B04
    green_band_idx = 2  # B03
    blue_band_idx = 1   # B02
    
    # 读取RGB波段数据
    red = src_ds.GetRasterBand(red_band_idx).ReadAsArray().astype(np.float32)
    green = src_ds.GetRasterBand(green_band_idx).ReadAsArray().astype(np.float32)
    blue = src_ds.GetRasterBand(blue_band_idx).ReadAsArray().astype(np.float32)
    
    # 检查数据范围并标准化
    max_value = np.max([np.max(red), np.max(green), np.max(blue)])
    print(f"原始数据最大值: {max_value}")
    
    # 如果数据未被除以10000，则进行标准化 - 固定使用10000作为缩放因子
    if max_value > 1.0:
        scale_factor = 10000.0  # Sentinel-2标准范围
        print(f"使用标准化因子: {scale_factor}")
        red = red / scale_factor
        green = green / scale_factor
        blue = blue / scale_factor
    
    # ------ 确保严格按照S2官方JavaScript实现 ------
    
    # 参数设置 - 直接使用官方脚本中的参数
    maxR = 3.0  # 最大反射率
    midR = 0.13  # 中间反射率点
    sat = 1.2  # 饱和度增强
    gamma = 1.8  # gamma校正
    
    # 裁剪函数 - 对应JavaScript中的clip函数
    def clip(s):
        return np.clip(s, 0, 1)
    
    # 对比度增强和高光压缩 - 对应JavaScript中的adj函数
    def adj(a, tx, ty, maxC):
        ar = clip(a / maxC)
        return ar * (ar * (tx/maxC + ty - 1) - ty) / (ar * (2 * tx/maxC - 1) - tx/maxC)
    
    # Gamma校正 - 对应JavaScript中的adjGamma函数
    gOff = 0.01
    gOffPow = np.power(gOff, gamma)
    gOffRange = np.power(1 + gOff, gamma) - gOffPow
    
    def adjGamma(b):
        return (np.power((b + gOff), gamma) - gOffPow) / gOffRange
    
    # 组合调整 - 对应JavaScript中的sAdj函数
    def sAdj(a):
        return adjGamma(adj(a, midR, 1, maxR))
    
    # 饱和度增强 - 对应JavaScript中的satEnh函数
    def satEnh(r, g, b):
        avgS = (r + g + b) / 3.0 * (1 - sat)
        return [clip(avgS + r * sat), clip(avgS + g * sat), clip(avgS + b * sat)]
    
    # sRGB转换 - 对应JavaScript中的sRGB函数
    def sRGB(c):
        return np.where(c <= 0.0031308, 
                      12.92 * c, 
                      1.055 * np.power(c, 0.41666666666) - 0.055)
    
    # ------ 应用处理链 - 完全按照evaluatePixel函数实现 ------
    
    # 1. 首先进行sAdj调整
    print("应用对比度增强和高光压缩...")
    red_adj = sAdj(red)
    green_adj = sAdj(green)
    blue_adj = sAdj(blue)
    
    # 2. 然后进行饱和度增强
    print("应用饱和度增强...")
    rgb_lin = satEnh(red_adj, green_adj, blue_adj)
    red_sat = rgb_lin[0]
    green_sat = rgb_lin[1]
    blue_sat = rgb_lin[2]
    
    # 3. 最后进行sRGB转换
    print("转换到sRGB空间...")
    red_srgb = sRGB(red_sat)
    green_srgb = sRGB(green_sat)
    blue_srgb = sRGB(blue_sat)
    
    # 转换为0-255的uint8
    red_uint8 = np.clip(red_srgb * 255, 0, 255).astype(np.uint8)
    green_uint8 = np.clip(green_srgb * 255, 0, 255).astype(np.uint8)
    blue_uint8 = np.clip(blue_srgb * 255, 0, 255).astype(np.uint8)
    
    # 创建目标文件
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(output_path, width, height, 3, gdal.GDT_Byte, 
                          options=["COMPRESS=LZW", "TILED=YES", "PHOTOMETRIC=RGB"])
    
    if dst_ds is None:
        print(f"错误: 无法创建输出文件 {output_path}")
        return
    
    # 设置空间参考信息
    dst_ds.SetGeoTransform(geo_transform)
    dst_ds.SetProjection(projection)
    
    # 写入RGB波段
    dst_ds.GetRasterBand(1).WriteArray(red_uint8)
    dst_ds.GetRasterBand(2).WriteArray(green_uint8)
    dst_ds.GetRasterBand(3).WriteArray(blue_uint8)
    
    # 设置波段颜色解释
    dst_ds.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
    dst_ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
    dst_ds.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
    
    # 计算统计信息
    for i in range(1, 4):
        dst_ds.GetRasterBand(i).ComputeStatistics(False)
    
    # 构建金字塔
    dst_ds.BuildOverviews("AVERAGE", [2, 4, 8, 16, 32])
    
    # 关闭数据集
    src_ds = None
    dst_ds = None
    
    print(f"TCI图像生成完成! 输出文件: {output_path}")
    print("注意: 此图像使用了与Sentinel-2官方完全相同的渲染算法")

if __name__ == "__main__":
    print("开始生成TCI彩色图像...")
    print("使用的Sentinel-2官方参数: maxR=3.0, midR=0.13, sat=1.2, gamma=1.8")
    create_tci_image(input_file, output_file)