from osgeo import gdal
from ATL_Tools import find_data_list,mkdir_or_exist
from tqdm import tqdm

# 打开输入图像
def convert_3857_to_4326(input_file, output_file):
    input_dataset = gdal.Open(input_file)

    # 获取输入图像的投影和地理参考信息
    input_projection = input_dataset.GetProjection()

    # 创建虚拟数据集（VRT）文件
    vrt_options = gdal.BuildVRTOptions(resampleAlg='cubic', srcNodata=None, VRTNodata=None)
    vrt_dataset = gdal.BuildVRT('/vsimem/temp.vrt', input_dataset, options=vrt_options)

    # 使用gdalwarp进行投影转换
    gdal.Warp(output_file, vrt_dataset, dstSRS="EPSG:4326")

    # 清理临时VRT数据集
    vrt_dataset = None
    gdal.Unlink('/vsimem/temp.vrt')


img_3857 = './Google3857—2/'
input_list = find_data_list(img_3857,'.tif')

img_4326 = './Google图源_4326/'

for img_3857_path in tqdm(input_list):

    out_put_path = img_3857_path.replace(img_3857, img_4326)
    convert_3857_to_4326(img_3857_path, out_put_path)

print('转换完成！')
