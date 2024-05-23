import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from ATL_Tools import mkdir_or_exist, find_data_list
from ATL_Tools.ATL_gdal import crop_tif_with_json_nan, Mosaic_all_imgs

import os
from osgeo import gdal
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def convert_24_to_12_classes(
        MASK_path_24: str,
        MASK_path_12: str):

    mkdir_or_exist(MASK_path_12)
    label_lists = find_data_list(MASK_path_24, suffix='.png')
    

    convert_dict=dict(
        class_1 = [1],
        class_2 = [2,3,4],
        class_3 = [5,6,7],
        class_4 = [9,10],
        class_5 = [11,13,14,15],
        class_6 = [12],
        class_7 = [18],
        class_8 = [17],
        class_9 = [16],
        class_10 = [8,19,20],
        class_11 = [21,22],
        class_12 = [23,24]
    )
    clsses_total = list(convert_dict.values())

    for mask_24_path in tqdm(label_lists):
        label_24 = np.array(Image.open(mask_24_path)).astype(np.uint8)
        label_12 = np.zeros_like(label_24)
        print(f"新创建图像的尺寸：{label_12.shape}")

        for i, classes_ in enumerate(clsses_total):
            print(f"===>大类别classes_:{classes_} ==> {i+1}")
            for j, classes__ in enumerate(classes_):
                print(f" -->小类classes__:{classes__}")
                label_12[label_24==classes__] = i+1
        Image.fromarray(label_12).save(os.path.join(MASK_path_12, os.path.basename(mask_24_path)))



def mask2RGB_with_Geoinfo(
        MASK_path: str,
        RGB_path: str,
        IMG_path: str):
    
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
    # 给生成的RGB图像添加空间
    add_meta_info = True 

    label_lists = find_data_list(MASK_path, suffix=label_suffix)

    METAINFO = dict(
        classes=("工业区域", "农田","树木",
                "草地", "水域","城市住宅",   
                "农村住宅", "裸地","雪",   
                "人造公园", "道路","火车站机场"),

        palette=[[200, 0, 0], [150, 250, 0], [0, 200, 0],
                [150, 200, 150], [0, 0, 200], [250, 0, 150],
                [200, 150, 150], [200, 200, 200], [250, 250, 250],
                [150, 150, 0], [250, 150, 0], [250, 200, 250]]
                )                                   

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

    # 1 convert 24 classes to 12 classes
    MASK_path_24 = '../推理结果-24类/推理结果-24-mask/'
    MASK_path_12 = '../推理结果-24类/推理结果-12-mask/'
    convert_24_to_12_classes(MASK_path_24, MASK_path_12)

    # 2 mask2RGB
    MASK_path = MASK_path_12
    RGB_path = '../推理结果-24类/推理结果-12-RGB/'
    IMG_path = '../要推理的images-矢量裁切/'
    mask2RGB_with_Geoinfo(MASK_path, RGB_path, IMG_path)

    # 3 Crop_RGB
    RGB_crop_path = '../推理结果-24类/推理结果-12-RGB-crop/'
    json_path = '../要推理的json/'
    crop_RGB_withGeo(RGB_path, RGB_crop_path, json_path)
 
    # 4 镶嵌RGB为一个大图
    out_path = '../推理结果-24类/24类-RGB-底图.tif'
    mosaic_RGB_withGeo(RGB_crop_path, out_path)
