# 本程序使用来裁切图像的，找到有目标的边框，并直接裁切

import cv2
import numpy as np
import os
from path import mkdir_or_exist, find_data_list
from PIL import Image
from tqdm import trange

img_root_path = r'D:\ATL\AI_work\ISAR\Dataset\SynISAR_low_resolution'
img_list = find_data_list(img_root_path)

out_root_path = r'D:\ATL\AI_work\ISAR\Dataset\SynISAR_no_background'
mkdir_or_exist(out_root_path)



def find_bbox(img_path):
    img_RGB = cv2.imread(img_path)
    img_2bit = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2GRAY)
    ret, img_2bit = cv2.threshold(src=img_2bit, thresh=19, maxval=255,type=cv2.THRESH_BINARY)

    x_list = []
    y_list = []

    for y in range(img_2bit.shape[0]):
        for x in range(img_2bit.shape[1]):
            
            # 如果像素为黑色（即值为0），更新最大最小坐标
            if img_2bit[y, x] != 0:
                x_list.append(x)
                y_list.append(y)
      
    return (min(x_list), min(y_list), max(x_list), max(y_list), img_2bit)

if __name__ == '__main__':
    for i in trange(len(img_list)):
        min_x, min_y, max_x, max_y, _ = find_bbox(img_list[i])
        img = cv2.imread(img_list[i])
        # new_img = np.zeros((max_y-min_y, max_x-min_x, 3))
        new_img = img[min_y:max_y, min_x:max_x, :]
        # print(min_x, min_y, max_x, max_y)
        # cv2.rectangle(img, (min_x, min_y), (max_x, max_y), color=(0, 0, 255)) # 画标签框
        cv2.imwrite(img_list[i].replace('SynISAR_low_resolution','SynISAR_no_background'), new_img)
