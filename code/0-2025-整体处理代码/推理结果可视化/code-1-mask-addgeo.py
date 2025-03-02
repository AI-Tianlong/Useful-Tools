import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from ATL_Tools import mkdir_or_exist, find_data_list
from ATL_Tools.ATL_gdal import crop_tif_with_json_nan, Mosaic_all_imgs

import os
from osgeo import gdal
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def mask_with_Geoinfo(
        MASK_path: str,
        MASK_GEO_path: str,
        IMG_path: str,
        backend='gdal'):
    
    """把mask转为RGB图像，并添加坐标信息

    Args:
        MASK_path (str): mask路径
        RGB_path (str): RGB图像路径
        IMG_path (str): 原始图像路径

    Returns:
        None, 生成的RGB图像保存在RGB_path
    """
    mkdir_or_exist(MASK_GEO_path)

    label_suffix = '.png'

    # 给生成的RGB图像添加坐标
    add_meta_info = True 

    label_lists = find_data_list(MASK_path, suffix=label_suffix)

    for mask_label_path in tqdm(label_lists, colour='Green'):
        mask_label = np.array(Image.open(mask_label_path)).astype(np.uint8)
        h,w = mask_label.shape

        if backend == 'gdal':
            output_path = os.path.join(MASK_GEO_path, os.path.basename(mask_label_path).replace(label_suffix, '.tif'))
            driver = gdal.GetDriverByName('GTiff')
            mask_geo_gdal = driver.Create(output_path, w, h, 1, gdal.GDT_Byte)

            mask_geo_gdal.GetRasterBand(1).WriteArray(mask_label)

            if add_meta_info:
                IMG_file_path = os.path.join(IMG_path, os.path.basename(mask_label_path).replace(label_suffix, '.tif'))
                IMG_gdal = gdal.Open(IMG_file_path, gdal.GA_ReadOnly)
                assert  IMG_gdal is not None, f"无法打开 {os.path.join(IMG_path, os.path.basename(mask_label_path).replace(label_suffix, '.tif'))}"

                trans = IMG_gdal.GetGeoTransform()
                proj = IMG_gdal.GetProjection()

                mask_geo_gdal.SetGeoTransform(trans)
                mask_geo_gdal.SetProjection(proj)

            mask_geo_gdal = None

if __name__ == '__main__':
    
    # 2 mask2RGB
    MASK_path = '../results_mask_24/'
    MASK_GEO_path = '../results_mask_24_geo/'
    img_path = '../S2-img/'
    mask_with_Geoinfo(MASK_path, MASK_GEO_path, img_path, backend='gdal')
