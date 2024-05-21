import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from ATL_Tools import mkdir_or_exist, find_data_list
import os
from osgeo import gdal
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

MASK_path = '/opt/AI-Tianlong/Datasets/ATL-ATLNongYe/ATL推理大图/黑河/推理结果-3类作物/推理结果-mask'
RGB_path = '/opt/AI-Tianlong/Datasets/ATL-ATLNongYe/ATL推理大图/黑河/推理结果-3类作物/推理结果-RGB'

IMG_path = '/opt/AI-Tianlong/Datasets/ATL-ATLNongYe/ATL推理大图/黑河/要推理的images-矢量裁切'
mkdir_or_exist(RGB_path)

label_suffix = '.png'

# 是否包含0类，classes和palette里没包含
reduce_zero_label = False
# 给生成的RGB图像添加空间
add_meta_info = True 

label_lists = find_data_list(MASK_path, suffix=label_suffix)

METAINFO = dict(
    classes=('Rice', 'Corn', 'soybean', 'Not-Farmland'),
    palette=[[0, 200, 250], [250, 200, 0], [150, 150, 250], [255, 255, 255]] # 绿色 黄色 粉色 白色 
    )                                                                    # 水稻 玉米 大豆 非农田

classes = METAINFO['classes']
palette = METAINFO['palette']

# palette = np.array(palette)

if reduce_zero_label:
    new_palette = [[0, 0, 0]] + palette
    print(f"palette: {new_palette}")
else:
    new_palette = palette
    print(f"palette: {new_palette}")

new_palette = np.array(new_palette)


for mask_label_path in tqdm(label_lists, colour='Green'):
    mask_label = np.array(Image.open(mask_label_path)).astype(np.uint8)
    h,w = mask_label.shape

    RGB_label = new_palette[mask_label]
    output_path = os.path.join(RGB_path, os.path.basename(mask_label_path).replace(label_suffix, '.tif'))

    driver = gdal.GetDriverByName('GTiff')
    RGB_label_gdal = driver.Create(output_path, w, h, 3, gdal.GDT_Byte)

    RGB_label_gdal.GetRasterBand(1).WriteArray(RGB_label[:,:,0])
    RGB_label_gdal.GetRasterBand(2).WriteArray(RGB_label[:,:,1])
    RGB_label_gdal.GetRasterBand(3).WriteArray(RGB_label[:,:,2])

    if add_meta_info:
        IMG_file_path = os.path.join(IMG_path, os.path.basename(mask_label_path).replace(label_suffix, '.tif'))
        IMG_gdal = gdal.Open(IMG_file_path, gdal.GA_ReadOnly)
        assert  IMG_gdal is not None, f"无法打开 {os.path.join(IMG_path, os.path.basename(mask_label_path).replace('.tif', '.tif'))}"

        trans = IMG_gdal.GetGeoTransform()
        proj = IMG_gdal.GetProjection()

        RGB_label_gdal.SetGeoTransform(trans)
        RGB_label_gdal.SetProjection(proj)

    RGB_label_gdal = None
