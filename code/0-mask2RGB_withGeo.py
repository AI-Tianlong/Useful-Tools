import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from ATL_Tools import mkdir_or_exist, find_data_list
from ATL_Tools.ATL_gdal import crop_tif_with_json_nan, Mosaic_all_imgs

import os
from osgeo import gdal
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def mask2RGB_with_Geoinfo(
        MASK_path: str,
        RGB_path: str,
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

    mkdir_or_exist(RGB_path)

    label_suffix = '.png'

    # 是否包含0类，classes和palette里没包含
    reduce_zero_label = True
    # 给生成的RGB图像添加坐标
    add_meta_info = True 

    label_lists = find_data_list(MASK_path, suffix=label_suffix)

    METAINFO = dict(
        classes=('耕地','林地','草地','水体',
                 '仓储工矿商业地，大面积的那种',
                 '住宅用地','公共设施',
                 '交通设施','裸地',
                 '冰川及积雪'),
        palette=[[150, 250, 0], [0, 150, 0], [250, 200, 0], [0, 100, 255],
                 [200, 0, 0],
                 [255, 217, 102],[250, 200, 150],
                 [250, 150, 0],[198, 89, 17],
                 [255, 255, 255]])                              
                    
    palette = METAINFO['palette']


    if reduce_zero_label:
        new_palette = [[0, 0, 0]] + palette
        print(f"palette: {new_palette}")
    else:
        new_palette = palette
        print(f"palette: {new_palette}")

    new_palette = np.array(new_palette)


    for mask_label_path in tqdm(label_lists, colour='Green'):
        mask_label = np.array(Image.open(mask_label_path)).astype(np.uint8)
        mask_label[mask_label==255]=0

        h,w = mask_label.shape

        RGB_label = new_palette[mask_label].astype(np.uint8)
        

        if backend == 'PIL':
            output_path = os.path.join(RGB_path, os.path.basename(mask_label_path).replace(label_suffix, '.png'))
            RGB_label = Image.fromarray(RGB_label).save(output_path)
        elif backend == 'gdal':
            output_path = os.path.join(RGB_path, os.path.basename(mask_label_path).replace(label_suffix, '.tif'))
            driver = gdal.GetDriverByName('GTiff')
            RGB_label_gdal = driver.Create(output_path, w, h, 3, gdal.GDT_Byte)

            RGB_label_gdal.GetRasterBand(1).WriteArray(RGB_label[:,:,0])
            RGB_label_gdal.GetRasterBand(2).WriteArray(RGB_label[:,:,1])
            RGB_label_gdal.GetRasterBand(3).WriteArray(RGB_label[:,:,2])

            if add_meta_info:
                IMG_file_path = os.path.join(IMG_path, os.path.basename(mask_label_path).replace(label_suffix, '.tif'))
                IMG_gdal = gdal.Open(IMG_file_path, gdal.GA_ReadOnly)
                assert  IMG_gdal is not None, f"无法打开 {os.path.join(IMG_path, os.path.basename(mask_label_path).replace(label_suffix, '.tif'))}"

                trans = IMG_gdal.GetGeoTransform()
                proj = IMG_gdal.GetProjection()

                RGB_label_gdal.SetGeoTransform(trans)
                RGB_label_gdal.SetProjection(proj)

            RGB_label_gdal = None

if __name__ == '__main__':
    
    # 2 mask2RGB
    MASK_path = './inference_result_mask_10/'
    RGB_path = './inference_result_RGB_with_Geo_10'
    img_path = './img/'
    mask2RGB_with_Geoinfo(MASK_path, RGB_path, img_path, backend='gdal')
