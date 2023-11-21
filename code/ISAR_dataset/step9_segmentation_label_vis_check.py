import os 
import numpy as np
from path import scandir, find_data_list, mkdir_or_exist
from tqdm import trange, tqdm
import cv2
from typing import List
from PIL import Image
from pandas import read_csv
import json
import imutils


real_img_path = '../Dataset_test/SynISAR_low_resolution'
RGB_img_path = '../Dataset_test/Segmentation_labels_120X120'

save_path = '../Dataset_test/Segmentation_labels_120X120_vis'
mkdir_or_exist(save_path)
for i in range(1,8):
    mkdir_or_exist(os.path.join(save_path, 'Aircraft-'+str(i)))

print('\n====================== Step 9 Create Segmentation labels vis ======================\n')
         
real_img_list = find_data_list(real_img_path, 'jpg')
RGB_img_list = find_data_list(RGB_img_path, '.png')

real_img_list = sorted(real_img_list)
RGB_img_list = sorted(RGB_img_list)


for index in trange(len(real_img_list)):
    # 底板图案
    bottom_pic = np.array(Image.open(real_img_list[index]).convert('RGB'))
    # Image.fromarray(bottom_pic).show()
    # 上层图案
    top_pic = np.array(Image.open(RGB_img_list[index]).convert('RGB'))
    # Image.fromarray(top_pic).show()
    # 权重越大，透明度越低
    overlapping = cv2.addWeighted(bottom_pic, 0.5, top_pic, 0.5, 0)
    # Image.fromarray(overlapping).show()
    
    # 保存叠加后的图片
    overlapping = cv2.cvtColor(overlapping, cv2.COLOR_BGR2RGB)
    cv2.imwrite(RGB_img_list[index].replace('Segmentation_labels_120X120', 'Segmentation_labels_120X120_vis'), overlapping)

print('\n================================= Done =================================\n')
         