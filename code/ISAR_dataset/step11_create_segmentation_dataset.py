# 本程序使用来裁切图像的，找到有目标的边框，并直接裁切

import cv2
import numpy as np
import os
from path import mkdir_or_exist, find_data_list
from PIL import Image
from tqdm import trange

# 根据img图像去检测bbox, 然后把同样的bbox 作用在mask上，应该可 

print('\n====================== Step 11 Create Segmentation Dataset ======================\n')

img_root_path = '../Dataset_test/SynISAR_no_background'
img_list = find_data_list(img_root_path)

mask_root_path = '../Dataset_test/Segmentation_labels_no_background'
mask_list = find_data_list(mask_root_path, '.png')

seg_dataset_root_path = '../Dataset_test/ATL_ISAR_Seg_dataset'
mkdir_or_exist(seg_dataset_root_path)

for i in ['train', 'val']:
    for j in ['images', 'labels']:
        for x in range(1,8):
            mkdir_or_exist(os.path.join(seg_dataset_root_path, i, j, 'Aircraft-'+str(x)))

# 从每个飞机类型 抽210 / 7 = 30张 作为测试集，剩下的为训练集 

for j in trange(1, 8):
    temp_image_dir = os.path.join(img_root_path,'Aircraft-'+str(j))
    temp_mask_dir = os.path.join(mask_root_path,'Aircraft-' + str(j))
    
    mask_path = find_data_list(temp_mask_dir,'.png')
    image_path = find_data_list(temp_image_dir,'jpg')
    mask_path = sorted(mask_path)
    image_path = sorted(image_path)
    for i in range(30):
        image = cv2.imread(image_path[i])
        labels = cv2.imread(mask_path[i])
        cv2.imwrite(os.path.join(seg_dataset_root_path, 'val' , 'images', 'Aircraft-'+str(j), str(i)+'.jpg'), image)
        cv2.imwrite(os.path.join(seg_dataset_root_path, 'val' , 'labels', 'Aircraft-'+str(j), str(i)+'.png'), labels)

    for x in range(31,181):
        image = cv2.imread(image_path[x])
        labels = cv2.imread(mask_path[x])
        cv2.imwrite(os.path.join(seg_dataset_root_path, 'train' , 'images', 'Aircraft-'+str(j), str(x)+'.jpg'), image)
        cv2.imwrite(os.path.join(seg_dataset_root_path, 'train' , 'labels', 'Aircraft-'+str(j), str(x)+'.png'), labels)