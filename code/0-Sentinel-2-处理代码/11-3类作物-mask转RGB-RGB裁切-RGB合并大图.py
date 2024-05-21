import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from ATL_Tools import mkdir_or_exist, find_data_list
from ATL_Tools.ATL_gdal import crop_tif_with_json_nan, Mosaic_all_imgs

import os
from osgeo import gdal
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def mask2RGB_with_Geoinfo(MASK_path: str, RGB_path: str, IMG_path: str):

    """把mask转为RGB图像，并添加坐标信息
    
    Args:
        MASK_path (str): mask路径
        RGB_path (str): RGB图像路径
        IMG_path (str): 原始图像路径

    Returns:
        None, 生成的RGB图像保存在RGB_path
    """

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


def crop_RGB_withGeo(
        RGB_path: str,
        output_path: str,
        json_path_all: str):

    mkdir_or_exist(output_path)
    img_list = find_data_list(RGB_path, suffix='.tif')

    for img_path in tqdm(img_list, colour='Green'):

        img_output_path = os.path.join(output_path, os.path.basename(img_path))
        json_path = os.path.join(json_path_all, os.path.basename(img_path).split('_')[-1].replace('.tif', '.json'))
        print(f'正在裁切: {img_output_path},json: {json_path}')
        crop_tif_with_json_nan(img_path, img_output_path, json_path)

def mosaic_RGB_withGeo(
        RGB_crop_path: str,
        output_path: str):

    Mosaic_all_imgs(img_file_path = RGB_crop_path, 
                    output_path = output_path, 
                    nan_or_zero='nan', 
                    output_band_chan=3,
                    add_alpha_chan = True) # 32位图像不支持alpha通道,改为False




if __name__ == '__main__':

    # 1 mask2RGB
    MASK_path = '../推理结果-3类/推理结果-3-mask/'
    RGB_path = '../推理结果-3类/推理结果-3-RGB/'
    IMG_path = '../要推理的images-矢量裁切/'
    mkdir_or_exist(RGB_path)
    mask2RGB_with_Geoinfo(MASK_path, RGB_path, IMG_path)

    # 2 Crop_RGB
    RGB_crop_path = '../推理结果-3类/推理结果-3-RGB-Crop/'
    json_path = '../要推理的json/'
    crop_RGB_withGeo(RGB_path, RGB_crop_path, json_path)
 
    # 3 镶嵌RGB为一个大图
    out_path = '../推理结果-3类/3类-RGB-底图.tif'
    mosaic_RGB_withGeo(RGB_crop_path, out_path)
