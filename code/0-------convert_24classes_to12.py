import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from ATL_Tools import mkdir_or_exist, find_data_list
import os
from osgeo import gdal
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

MASK_path_24 = '/opt/AI-Tianlong/Datasets/ATL_DATASETS/Harbin/Harbin_big_labels'
MASK_path_12 = '/opt/AI-Tianlong/Datasets/ATL_DATASETS/Harbin/Harbin_big_labels_12classes'

mkdir_or_exist(MASK_path_12)

label_lists = find_data_list(MASK_path_24, suffix='.tif')

class_1 = [1]
class_2 = [2,3,4]
class_3 = [5,6,7]
class_4 = [9,10]
class_5 = [11,13,14,15]
class_6 = [12]
class_7 = [18]
class_8 = [17]
class_9 = [16]
class_10 = [8,19,20]
class_11 = [21,22]
class_12 = [23,24]
clsses_total = [class_1,class_2,class_3,class_4,class_5,class_6,class_7,class_8,class_9,class_10,class_11,class_12]

for mask_24_path in tqdm(label_lists):
    label_24 = np.array(Image.open(mask_24_path)).astype(np.uint8)

    label_12 = np.zeros_like(label_24)
    print(f"新创建图像的尺寸：{label_12.shape}")

    for i, classes_ in enumerate(clsses_total):
        print(f"=>大类别classes_:{classes_} ==> {i+1}")
        for j, classes__ in enumerate(classes_):
            print(f"===>小类classes__:{classes__}")
            label_12[label_24==classes__] = i+1
    Image.fromarray(label_12).save(os.path.join(MASK_path_12, os.path.basename(mask_24_path).replace('.tif', '.tif')))
