from mimetypes import suffix_map
from ATL_Tools import mkdir_or_exist, find_data_list
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
from tqdm import tqdm
import copy
from osgeo import gdal
from typing import List, Dict, Optional


def convert_19_to_10_classes(
        old_label_path: str,
        new_label_save_path: str,
        convert_dict: Optional[Dict]=None,
        class_drop: Optional[List]=None,
        with_geo=True):  # 去掉类别16,这一类的名字是固定的):
    """把24类别的标签转为5类别的标签

    Args:
        old_label_path (str): 24类别的标签路径
        new_label_save_path (str): 5类别的标签保存路径
        convert_dict (Dict, optional): 24类别到5类别的转换字典. Defaults to None.
        class_drop (List, optional): 要去掉的类别. Defaults to None,指原标签中的.

    Examples:
        >>> # 24类别到5类别的转换字典
        >>> convert_dict=dict(
                class_0_Invalid = [0],
                class_1_Vegetation = [2, 3, 4, 5, 6, 7, 9, 10],
                class_2_water = [11, 13, 14, 15],
                class_3_Manmade = [8, 12, 18, 19, 20, 21, 22, 23, 24],
                class_4_BareLand = [17],
                class_5_snow = [16],
            )
        >>> class_drop = [16]

        >>> # 24类别到12类别的转换字典
        >>> convert_dict=dict(
                class_0 = [0],
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
    """

    if convert_dict == None:
        assert False, "convert_dict is None, 请指定, 示例查看函数说明"        
    
    clsses_list = list(convert_dict.values())

    old_label_ds = gdal.Open(old_label_path)
    old_label_np = old_label_ds.ReadAsArray()
    new_label_np = copy.deepcopy(old_label_np)
    h, w = new_label_np.shape[0], new_label_np.shape[1]

    # 创建 convert 后的图像
    driver = gdal.GetDriverByName('GTiff')
    new_label_ds = driver.Create(new_label_save_path, w, h, 1, gdal.GDT_Byte)

    for i, classes_ in enumerate(clsses_list):
        print(f" 【ATL-LOG】合并 ==> {classes_} ==> {i}")
        for j, classes__ in enumerate(classes_):
            new_label_np[old_label_np==classes__] = i
            print(f"            {classes__} --> {i} Done")

    if class_drop != None:
        for class_drop_ in class_drop:
            new_label_np[old_label_np==class_drop_] = 255
        
        # print(f" 【ATL-LOG】丢弃 ==> {class_drop_} ==> 255 Done")


    new_label_ds.GetRasterBand(1).WriteArray(new_label_np)
    if with_geo:
        trans = old_label_ds.GetGeoTransform()
        proj = old_label_ds.GetProjection()

        new_label_ds.SetGeoTransform(trans)
        new_label_ds.SetProjection(proj)

        old_label_ds = None
        new_label_ds = None
    
if __name__ == '__main__':
    
    Old_Label_Path = '../results_mask_18_geo/'
    New_Label_Save_Path = '../results_mask_10_geo/'

    suffix = '.tif'

    mkdir_or_exist(New_Label_Save_Path)

    # convert_dict=dict(
    #     class_0_Invalid = [0],
    #     class_1_Vegetation = [2, 3, 4, 5, 6, 7, 9, 10],
    #     class_2_water = [11, 13, 14, 15],
    #     class_3_Manmade = [8, 12, 18, 19, 20, 21, 22, 23, 24],
    #     class_4_BareLand = [17],
    #     class_5_snow = [16],
    # )
    # class_drop = [16]


    convert_dict=dict(
        class_0 = [0],
        class_1 = [1,2],
        class_2 = [3],
        class_3 = [4,5],
        class_4 = [6,7,8],
        class_5 = [9],
        class_6 = [10,11],
        class_7 = [12,13],
        class_8 = [14, 15,16,17],
        class_9 = [18],
        class_10 = [19],
    
    )

    # class_drop = [16] # 原标签中的 雪 是16,扔掉 
    class_drop = [] # 原标签中的 雪 是16,扔掉 

    old_label_list = find_data_list(Old_Label_Path, suffix=suffix)

    for old_label_path_ in tqdm(old_label_list, colour='Green'):
        new_label_save_path = os.path.join(New_Label_Save_Path, os.path.basename(old_label_path_).replace('_18label'+suffix, '_10label'+suffix))
        convert_19_to_10_classes(old_label_path_, new_label_save_path, convert_dict, class_drop, with_geo = True)



    