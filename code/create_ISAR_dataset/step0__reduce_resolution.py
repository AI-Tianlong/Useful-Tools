# Author: Tianlong Ai
# Github-ID: AI-Tianlong

import os 
import numpy as np
from path import scandir
from tqdm import trange, tqdm
import cv2

img_root_path = 'D:/ATL/AI_work/ISAR/Dataset/SynISAR_original'

def find_data_path(img_root_path):
    print('\n==============================================================')
    print('-- 正在读取数据集列表...')

    img_list = []
    for img_name in scandir(img_root_path, suffix='.jpg', recursive=True):
        if '.jpg' in img_name:
            img_path = os.path.join(img_root_path, img_name)
            img_list.append(img_path)
    print(f'-- 共在 {img_root_path} 下寻找到图片 {len(img_list)} 张')

    return img_list

def reduce_resolution(img_list):
    for img_path in tqdm(img_list):
        img = cv2.imread(img_path)
        resize_image = cv2.resize(img, (0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(img_path.replace('SynISAR_original', 'SynISAR_low_resolution'), resize_image)
    print('-- 已完成...')
    print('==============================================================\n')


def main():
    img_list = find_data_path(img_root_path)
    reduce_resolution(img_list)


if __name__ == '__main__':
    main()