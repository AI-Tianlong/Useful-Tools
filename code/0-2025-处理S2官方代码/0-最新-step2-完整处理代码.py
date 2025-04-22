from ATL_Tools import find_data_list, mkdir_or_exist
import os
from osgeo import gdal, ogr, osr
import glob
from tqdm import tqdm 
from ATL_Tools.ATL_gdal import crop_tif_with_json_zero, save_array_to_tif, read_img_get_geo, align_image
import subprocess
import shutil
import os
import math
import numpy as np
from typing import List

# 读取文档4中的文件，第一列为S2的文件名，第二三列为图像名
def read_txt_to_dict(file_path):
    result_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 去掉换行符并按逗号分割
            parts = line.strip().split(',')
            if len(parts) > 1:
                key = parts[0]
                values = parts[1:]
                result_dict[key] = values
    return result_dict

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
    data_type = gdal.GDT_UInt16
    
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
    
    # print(f"合并完成! 输出文件: {output_path}")
    # print("注意: 所有像素值已除以10000并保存为float32格式")


def delete_folder_with_terminal(folder_path):
    """
    使用终端命令删除文件夹
    :param folder_path: 要删除的文件夹路径
    """
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return

    try:
        # 判断操作系统
        if os.name == 'nt':  # Windows 系统
            # 使用 rmdir /s /q 删除文件夹
            # subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", folder_path], check=True)
            subprocess.run(["rm", "-rf", folder_path], check=True)
        else:  # Linux 或 macOS
            # 使用 rm -rf 删除文件夹
            subprocess.run(["rm", "-rf", folder_path], check=True)
        print(f"成功删除文件夹: {folder_path}")
    except subprocess.CalledProcessError as e:
        print(f"删除文件夹失败: {folder_path}, 错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

  
def Mosaic_all_imgs(img_file_path: str,
                    output_path: str,
                    add_alpha_chan: bool=False,
                    nan_or_zero: str='zero',
                    output_band_chan: int=None,
                    img_list:List[str]=None,
                    band_order_list:List[str]=None) -> None:
    
    # Step1：如果两张图的尺寸不一样的话，先将两张图进行像素对齐，然后再mosaic。
    # Step2：将两张图进行镶嵌，镶嵌的方式是：将小图的像素值加到大图上去
    '''✔将指定路径文件夹下的tif影像全部镶嵌到一张影像上
        细节测试:
        mosaic图像：
        ✔ 镶嵌根据矢量裁切后的图像，无数据区域为nan
        ✔ 镶嵌矩形的图像，无数据区域为nan
        mosaic标签：
        x 彩色标签是uint8的，不能用nan，只能用0


    注：将多个图合并之后，再进行裁切的话，nan就是白的，zero就是黑的
        如果单独裁切一个的话，不管nan还是zero都是白的
        如果将裁切后的进行合并的话，会把nan的部分也合并进去，需要单独处理

        需要进行优化
    
    Args:
        img_file_path (str)：tif 影像存放路径
        output_path (str): 输出镶嵌后 tif 的路径
        add_alpha_chan (bool): 是否添加alpha通道，将无数据区域显示为空白，默认为False
        Nan_or_Zero (str): 'nan'或'zero'镶嵌后的无效数据nan或0,默认为0
                           'nan'更适合显示，'0更适合训练'
        output_band_chan (int): 对于多光谱图像，如果只想保存前3个通道的话，指定通道数
        img_list (List[str]):  需要镶嵌的影像列表，默认为None
        
        例子: Mosaic_all_imgs(img_path_all, output_path, add_alpha_chan=True) # 对于RGB标签，添加alpha通道
              Mosaic_all_imgs(img_path_all, output_path, Nan_or_Zero='zero') # 对于float32 img，mosaic为zero
              Mosaic_all_imgs(img_path_all, output_path, Nan_or_Zero='zero') # 对于float32 img，mosaic为nan #展示用

    Returns: 
        镶嵌合成的整体影像
    '''

    # os.chdir(img_file_path) # 切换到指定路径
    #如果存在同名影像则先删除
    if os.path.exists(output_path):
        print(f"  【ATL-LOG】存在{output_path}, 已覆盖")
        os.remove(output_path)

    # 如果不指定的话，就找所有的tif文件
    if img_list == None:
        print("  【ATL-LOG】未指定img_list, 寻找所有tif文件...")
        all_files = find_data_list(img_file_path, suffix='.tif') # 寻找所有tif文件
       
    elif img_list != None:
        print("  【ATL-LOG】指定img_list, 寻找指定的tif文件...")
        all_files = img_list
    assert all_files!=None, 'No tif files found in the path'
    print(f"  【ATL-LOG】本次镶嵌的影像有{len(all_files)}张")

    # 获取所有列表中，影像的像素尺寸，如果尺寸不一致的话，先进行像素级的对齐，然后再合并。合并按照最大的尺寸去合并。
    need_align_flag = False
    file_0_ds = gdal.Open(all_files[0])
    max_h, max_w =  file_0_ds.RasterYSize, file_0_ds.RasterXSize
    # import pdb; pdb.set_trace()
    for in_fn in all_files:
        in_ds = gdal.Open(in_fn)
        if in_ds is None:
            assert False, f'  【ATL-LOG】打开影像失败: {in_fn}, 请检查路径是否正确'
        print(f'  【ATL-LOG】读入图像的尺寸：{in_ds.RasterYSize, in_ds.RasterXSize} (高，宽)')
        # print(f'  【ATL-LOG】读入图像的尺寸：{in_ds.RasterXSize, in_ds.RasterYSize} (宽，高)')
        img_h_, img_w_ = in_ds.RasterYSize, in_ds.RasterXSize
        
        if img_h_ != max_h or img_w_ != max_w:
            need_align_flag = True
            max_h = max(img_h_, max_h)
            max_w = max(img_w_, max_w)   
    

    if need_align_flag:
        print(f'  【ATL-LOG】镶嵌的影像尺寸不一致，请先进行像素对齐')
        # 这里需要进行像素对齐了，先对齐到最大值
        for in_fn in all_files:
            in_ds = gdal.Open(in_fn)
            if in_ds is None:
                assert False, f'  【ATL-LOG】打开影像失败: {in_fn}, 请检查路径是否正确'
            print(f'  【ATL-LOG】读入图像的尺寸：{in_ds.RasterYSize, in_ds.RasterXSize} (高，宽)')
            img_h_, img_w_ = in_ds.RasterYSize, in_ds.RasterXSize
            out_img_path = os.path.join(os.path.dirname(in_fn), os.path.basename(in_fn).replace('.tif', '_aligned.tif'))
            align_image(src_path=in_fn, ref_path=all_files[0], src_out_put_path=out_img_path)
            all_files[all_files.index(in_fn)] = out_img_path

    # 替换掉all_files中的文件名，替换成对齐后的文件名
    # import pdb; pdb.set_trace()
    # 最终的镶嵌影像的尺寸、pr

    #获取待镶嵌栅格的最大最小的坐标值，# 这里为最后的镶嵌影像的地理坐标
    min_x, min_y, max_x, max_y = read_img_get_geo(all_files[0])  # 越往下，维度越低，约大
    for in_fn in all_files:
        minx, maxy, maxx, miny = read_img_get_geo(in_fn)
        min_x = min(min_x, minx)
        min_y = min(min_y, miny)
        max_x = max(max_x, maxx)
        max_y = max(max_y, maxy)


    #计算镶嵌后影像的行列号
    in_ds=gdal.Open(all_files[0])
    geotrans=list(in_ds.GetGeoTransform())

    # 这一行代码获取了数据集的地理转换信息，包括地理坐标系的变换参数。
    # in_ds.GetGeoTransform() 返回一个包含六个浮点数的元组，
    # 分别表示左上角的X坐标、水平像素分辨率、X方向的旋转（通常为0）、
    # 左上角的Y坐标、Y方向的旋转（通常为0）、垂直像素分辨率。
    # 这一行代码将获取的元组转换为列表形式，并赋值给 geotrans。

    width_geo_resolution = geotrans[1]
    heigh_geo_resolution = geotrans[5] # 负值
    # print(f'width_geo_resolution:{width_geo_resolution}, heigh_geo_resolution:{heigh_geo_resolution}')

    # columns = math.ceil((max_x-min_x) / width_geo_resolution) + 50 # 有几个像素的偏差，所以会导致小图超出大图范围！！！  # 不能加
    # rows = math.ceil((max_y-min_y) / (-heigh_geo_resolution)) + 50 # 有几个像素的偏差，所以会导致小图超出大图范围！！！ 

    columns = math.ceil((max_x-min_x) / width_geo_resolution)  # 有几个像素的偏差，所以会导致小图超出大图范围！！！  # 不能加
    rows = math.ceil((max_y-min_y) / (-heigh_geo_resolution))  # 有几个像素的偏差，所以会导致小图超出大图范围！！！ 

    bands = in_ds.RasterCount  # 读进来的bands
    if not output_band_chan==None:
        bands = output_band_chan
        print("  【ATL-LOG】人为指定输出波段数为:", bands)
    print(f'  【ATL-LOG】新合并图像的尺寸: {rows, columns, bands} (高，宽，波段)')

    in_band_DataType = in_ds.GetRasterBand(1).DataType

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, columns, rows, bands, in_band_DataType)
    out_ds.SetProjection(in_ds.GetProjection())
    geotrans[0] = min_x
    geotrans[3] = max_y
    out_ds.SetGeoTransform(geotrans)


    #定义仿射逆变换
    inv_geotrans = gdal.InvGeoTransform(geotrans)

    #开始逐渐写入
    for in_fn in tqdm(all_files, desc='正在镶嵌图像ing...', colour='GREEN'):
        print('正在镶嵌:', os.path.abspath(in_fn))
        in_ds = gdal.Open(in_fn)
        if in_ds is None:
            assert False, f'  【ATL-LOG】打开影像失败: {in_fn}, 请检查路径是否正确'
        in_gt = in_ds.GetGeoTransform()
        # print(f'  【ATL-LOG】读入图像的尺寸：{in_ds.RasterYSize, in_ds.RasterXSize} (高，宽)')

        # import pdb; pdb.set_trace()
        #仿射逆变换
        offset = gdal.ApplyGeoTransform(inv_geotrans, in_gt[0], in_gt[3])
        x, y = map(int, offset)
        # print(f'逆变换后的像素：x:{x}, y:{y}')
        # 该函数返回一个转换器对象 trans，可以使用这个对象执行从输入数据集到输出数据集的坐标转换
        trans = gdal.Transformer(in_ds, out_ds, [])       # in_ds是源栅格，out_ds是目标栅格
        success, xyz = trans.TransformPoint(False, 0, 0)  # 计算in_ds中左上角像元对应out_ds中的行列号
        x, y, z = map(int, xyz)
        # print(f'  【ATL-LOG】小图(0, 0)变换到大图的像素：(({y},{x}), ({y+in_ds.RasterYSize},{x+in_ds.RasterXSize}))')

        for band_num in range(bands):
            # 小图的单通道，(h,w),无数据全是nan
            in_ds_array = in_ds.GetRasterBand(band_num + 1).ReadAsArray() #(h,w)
            # 无效数据的地方全是nan,这也符合下载下来的图，空缺的地方是nan、
            # 把nan的地方，用0替代
            in_ds_array = np.nan_to_num(in_ds_array, nan=0)

            # 最后合并的大图的波段通道，(h,w)，没数据的地方全是0
            big_out_band = out_ds.GetRasterBand(band_num + 1)
            # 大图中，小图区域的数据
            Tiny_in_BigOut_data = big_out_band.ReadAsArray(x, y, in_ds_array.shape[1], in_ds_array.shape[0])
            Tiny_in_BigOut_data = np.nan_to_num(Tiny_in_BigOut_data, nan=0)
            # 最后要写入大图的数据：如果是根据矢量裁切完的应该不会有重合，直接相加就行
            # 但是如果是矩形大图的话，有重叠的话，则需要舍弃小图的重叠部分。
            # 第一步：找到大图中有数据的区域：即大图不为零的地方
            # 第二步：利用大图中不为零的地方，把小图中的值设置为0 
            # 第三部：把两个图相加，得到最后的结果
            zero_mask_in_tiny_of_big = Tiny_in_BigOut_data!=0
            in_ds_array[zero_mask_in_tiny_of_big] = 0
            # print(f'小图的尺寸{in_ds_array.shape}')
            # print(f'大图中小图尺寸：{Tiny_in_BigOut_data.shape}')
            
            in_ds_array = Tiny_in_BigOut_data + in_ds_array

            # 写入大图
            big_out_band.WriteArray(in_ds_array, x, y)
            if band_order_list is not None:
                big_out_band.SetDescription(band_order_list[band_num]) 
    del in_ds, out_ds # 必须要有这个

    if nan_or_zero == 'zero' and add_alpha_chan == False:
        print(f"  【ATL-LOG】空缺部分为'zero', 不添加alpha通道, 支持float32-img、uint8-label")
        pass
    # 最后把所有为0.的地方都变成nan
    # 如果是float32图像的话,nan是可以work的,则会让无数据的地方变成nan,显示的时候就是透明的
    elif nan_or_zero == 'nan' and add_alpha_chan ==  False:
        print(f"  【ATL-LOG】空缺部分为'nan', 不添加alpha通道,支持-float32img")
        output_img_ds = gdal.Open(output_path)
        Transform = output_img_ds.GetGeoTransform()
        Projection = output_img_ds.GetProjection()
        
        img_nan = output_img_ds.ReadAsArray()
        img_nan = img_nan.transpose(1,2,0)
        img_nan[img_nan==0.] = np.nan

        save_array_to_tif(img_array = img_nan,
                          out_path = output_path, # 覆盖图像
                          Transform = Transform,
                          Projection = Projection,
                          Datatype = output_img_ds.GetRasterBand(1).DataType,
                          Band = bands)
        
    print(f'-->镶嵌图像已完成，输出至 {output_path}')
     


def crop_tif_with_json_zero_new(img_path,
                            output_path: str,
                            geojson_path: str,
                            nodata_value: int = 255):
  
    if os.path.exists(output_path):
        print(f"--> 存在{output_path}, 已覆盖")
        os.remove(output_path)

    # 打开栅格文件
    if isinstance(img_path, str):
        raster_ds = gdal.Open(img_path)
    elif isinstance(img_path, gdal.Dataset):
        raster_ds = img_path
    assert raster_ds!=None, f'打开 {raster_ds} 失败'

    # 打开GeoJSON文件
    geojson_ds = ogr.Open(geojson_path)
    geojson_layer = geojson_ds.GetLayer()

    # 获取GeoJSON文件的范围
    xmin, xmax, ymin, ymax = geojson_layer.GetExtent()
    # 设置裁剪范围
    # 设置投影为图像的投影

    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(raster_ds.GetProjection())
    epsg_code = spatial_ref.GetAuthorityCode(None) # '32651'
    if epsg_code:
        dst_srs = f'EPSG:{epsg_code}'
    else:
        raise ValueError("无法确定图像的EPSG代码，请检查图像投影！")

    # import pdb; pdb.set_trace()

    warp_options = gdal.WarpOptions(cutlineDSName=geojson_path,
                                    cutlineWhere=None,
                                    cropToCutline=True, # 裁剪到矢量范围
                                    dstNodata = nodata_value, # 设置裁剪后的无数据值为 255
                                    outputBounds=(xmin, ymin, xmax, ymax),
                                    dstSRS=dst_srs) # EPSG:32651
                                    # dstSRS='EPSG:4326')  # 设置输出投影，这里使用EPSG:4326，即WGS84经纬度坐标系


    # 执行裁剪
    gdal.Warp(output_path, raster_ds, options=warp_options)

    # 关闭数据源
    raster_ds = None
    geojson_ds = None
    if isinstance(img_path, str):
        print(f'--> 根据矢量裁切{img_path}完成！无数据区域设置为 {nodata_value}')
    elif isinstance(img_path, gdal.Dataset):
        print(f'--> 根据矢量裁切完成！无数据区域设置为 {nodata_value}')

def intersect(ext1, ext2):
    min_x = max(ext1[0], ext2[0])  # 计算交集的最小x坐标
    min_y = max(ext1[1], ext2[1])  # 计算交集的最小y坐标
    max_x = min(ext1[2], ext2[2])  # 计算交集的最大x坐标
    max_y = min(ext1[3], ext2[3])  # 计算交集的最大y坐标
    if (min_x > max_x) or (min_y > max_y):
        return None  # 如果无交集，返回None
    else:
        return [min_x, min_y, max_x, max_y]  # 返回交集坐标
def align_image_new(src_path: str, 
                ref_path: str, 
                src_out_put_path: str,  
                src_resampleAlg=gdal.GRA_Bilinear):
    """把 src_path 对齐到 dst_path，并保存到 out_put_path
    把 src 的分辨率、范围、像素对齐到dst
    Args：
        
    """

    src_ds = gdal.Open(src_path)
    src_trans = src_ds.GetGeoTransform()  # 获取源影像的地理变换参数
    src_extent = [src_trans[0],
                  src_trans[3] + src_trans[5] * src_ds.RasterYSize,
                  src_trans[0] + src_trans[1] * src_ds.RasterXSize,
                  src_trans[3]]  # 计算源影像的范围
    
    ref_ds = gdal.Open(ref_path)
    ref_trans = ref_ds.GetGeoTransform()  # 获取目标影像的地理变换参数
    ref_extent = [ref_trans[0],
                  ref_trans[3] + ref_trans[5] * ref_ds.RasterYSize,
                  ref_trans[0] + ref_trans[1] * ref_ds.RasterXSize,
                  ref_trans[3]]  # 计算目标影像的范围

    bound = intersect(src_extent, ref_extent)  # 计算源影像和目标影像的交集范围
    print(f'当前图片 basename{src_path} bound:', bound)
    ref_resx = ref_trans[1]  # 获取目标x方向分辨率
    ref_resy = ref_trans[5]  # 获取目标y方向分辨率


    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')  # 设置GDAL使用所有CPU线程

    align_src_option = gdal.WarpOptions(
                    format='GTiff', outputBounds=bound,  # 设置输出格式为VRT，输出范围为交集范围
                    xRes=ref_resx, yRes=ref_resy,              # 设置x和y方向的分辨率
                    dstNodata=0, srcNodata=0, srcAlpha=False,  # 设置无数据值
                    resampleAlg=src_resampleAlg,     # 设置重采样算法为最近邻插值
                    multithread=True,                  # 启用多线程
                    warpOptions=['GDAL_NUM_THREADS=ALL_CPUS']  # 设置GDAL使用所有CPU线程
                    )
    align_src = gdal.Warp(src_out_put_path, srcDSOrSrcDSTab=src_ds, options=align_src_option)  # 对目标影像进行重投影和对齐
    print('')

if __name__ == "__main__":
        
    name_txt_file_path = './文档-4-S2-图片名称和S2文件名称对应列表.txt'
    big_img_folder_path = '../0-25年官网下载2019年数据-大图文件/'
    crop_save_dir = '../0-25年官网下载2019年数据-按照json切片/'
    final_s2_dataset_save_path = '../0-25年官网下载2019年数据-最终数据集文件/'
    json_file_path = '../0-需要下载的影像的json矢量文件/'
    mkdir_or_exist(crop_save_dir)
    mkdir_or_exist(final_s2_dataset_save_path)

    # 调用函数并打印结果
    data_dict = read_txt_to_dict(name_txt_file_path)
    for key, value in data_dict.items():
        print(f"{key}: {value}")

    # 开始按照键值对大图进行裁切
    for key, value in data_dict.items():
        crop_img_name = key  # s2数据集的文件名
        
        if os.path.exists(os.path.join(crop_save_dir, crop_img_name)):
            print(f'文件夹已存在，跳过裁切：{crop_img_name}')
            continue

        for big_img_dir_name in value:
            big_img_dir_path = os.path.join(big_img_folder_path, big_img_dir_name+'-大图')
            # import pdb; pdb.set_trace()
            crop_img_save_dir = os.path.join(crop_save_dir, crop_img_name, big_img_dir_name+'-未像素对齐')
            mkdir_or_exist(crop_img_save_dir)
            to_crop_img_list = find_data_list(big_img_dir_path, '.jp2')
            # step0: 创建空文件夹
            mkdir_or_exist(os.path.join(crop_save_dir, crop_img_name))

            # step1: 按照json矢量裁切大图，并保持原始图像的投影和分辨率
            for to_crop_img_path in to_crop_img_list:
                crop_img_save_path = os.path.join(crop_img_save_dir, os.path.basename(to_crop_img_path).replace('.jp2', '.tif'))
                crop_json_path = os.path.join(json_file_path, crop_img_name + '.json')
                crop_tif_with_json_zero_new(to_crop_img_path, crop_img_save_path, crop_json_path, nodata_value=0)
            
            # step2：像素对齐
            align_crop_img_save_dir = os.path.join(crop_save_dir, crop_img_name, big_img_dir_name+'-像素对齐')
            mkdir_or_exist(align_crop_img_save_dir)
            ref_img_path = find_data_list(crop_img_save_dir, suffix='_B02_10m.tif')[0] # 只有一个符合，按照这个当做参考
            to_align_img_list = find_data_list(crop_img_save_dir, suffix='.tif') # 只对齐20m的 6个
            for to_align_img_path in tqdm(to_align_img_list):
                to_align_img_basename = os.path.basename(to_align_img_path)
                if '_20m.tif' in to_align_img_basename:
                    align_img_output_path = os.path.join(align_crop_img_save_dir, to_align_img_basename.replace('20m.tif', '10m.tif'))
                    # 把对齐像素后的 图像和标签 都保存出来
                    align_image_new(src_path=to_align_img_path, 
                                    ref_path=ref_img_path, 
                                    src_out_put_path = align_img_output_path)
                elif '_10m.tif' in to_align_img_basename:
                    align_img_output_path = os.path.join(align_crop_img_save_dir, to_align_img_basename.replace('10m.tif', '10m.tif'))
                    # 单纯的把文件复制一份
                    shutil.copy(to_align_img_path, align_img_output_path)
            
            # step3: 合并多波段图像
            band_order_list = ['B02', 'B03', 'B04', 'B05','B06','B07','B08', 'B8A', 'B11', 'B12']
            crop_img_10m_output_path = os.path.join(crop_save_dir, crop_img_name, big_img_dir_name + '_10bands_merged.tif')
            merge_bands(align_crop_img_save_dir, crop_img_10m_output_path, band_order_list)
            # 删除掉 step1 和 step2 的中间文件，节省空间
            # delete_folder_with_terminal(crop_img_save_dir)
            # delete_folder_with_terminal(align_crop_img_save_dir)

        # Step4：合并多张波段merge后的图像
        # 如果是由有多张图像组成的，则合并多张图像为一张最终的结果，并命名为S2的那个文件
        final_s2_img_save_path = os.path.join(crop_save_dir, crop_img_name, crop_img_name + '.tif')
        to_do_mosaic_list = find_data_list(os.path.join(crop_save_dir, crop_img_name), suffix='_10bands_merged.tif', recursive=False)
        if len(to_do_mosaic_list)>2:
            Mosaic_all_imgs('', final_s2_img_save_path, nan_or_zero='zero', img_list=to_do_mosaic_list, band_order_list=band_order_list)
            final_final_s2_img_save_path = os.path.join(final_s2_dataset_save_path, crop_img_name + '.tif')
            shutil.copy(final_s2_img_save_path, final_final_s2_img_save_path)
        else:
        # step5: 保存一份最终的S2图像到新文件夹中
            final_final_s2_img_save_path = os.path.join(final_s2_dataset_save_path, crop_img_name + '.tif')
            shutil.copy(to_do_mosaic_list[0], final_final_s2_img_save_path)



