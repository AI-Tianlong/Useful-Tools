# Author: Tianlong Ai
# Github-ID: AI-Tianlong

import os 
import numpy as np
from path import mkdir_or_exist, find_data_list
from tqdm import trange, tqdm
import cv2

img_root_path = '../Dataset_test/SynISAR_original'

def reduce_resolution(img_list):
    for img_path in tqdm(img_list):
        img = cv2.imread(img_path)
        resize_image = cv2.resize(img, (0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(img_path.replace('SynISAR_original', 'SynISAR_low_resolution'), resize_image)
    print('-- 已完成...')
    print('==============================================================\n')


def main():
    img_list = find_data_list(img_root_path, suffix='jpg')
    mkdir_or_exist(img_root_path.replace('SynISAR_original', 'SynISAR_low_resolution'))
    for i in range(1,8):
        mkdir_or_exist(os.path.join(
            img_root_path.replace('SynISAR_original', 'SynISAR_low_resolution'),
            'Aircraft-'+str(i)))
    reduce_resolution(img_list)


if __name__ == '__main__':
    main()