# 本程序使用来裁切图像的，找到有目标的边框，并直接裁切

import cv2
import numpy as np
import os
from path import mkdir_or_exist, find_data_list
from PIL import Image
from tqdm import trange

# 根据img图像去检测bbox, 然后把同样的bbox 作用在mask上，可
# 只能运行一次，不然会有问题，出问题，重新运行 step 8 -> step 9 -> step 10 

print('\n====================== Step 10 Crop Segmentation labels ======================\n')

img_root_path = '../Dataset_test/SynISAR_low_resolution'
img_list = find_data_list(img_root_path)
img_list = sorted(img_list)

mask_root_path = '../Dataset_test/Segmentation_labels_120X120'
mask_list = find_data_list(mask_root_path, '.png')
mask_list = sorted(mask_list)


out_root_path = '../Dataset_test/Segmentation_labels_no_background'
for i in range(1, 8):
    mkdir_or_exist(os.path.join(out_root_path, 'Aircraft-'+str(i)))

def find_bbox(img_path):
    img_RGB = cv2.imread(img_path)
    img_2bit = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2GRAY)
    ret, img_2bit = cv2.threshold(src=img_2bit, thresh=19, maxval=255,type=cv2.THRESH_BINARY)
    # cv2.imshow('img', img_2bit)
    # cv2.waitKey(0)
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
        min_x, min_y, max_x, max_y, origin_img = find_bbox(img_list[i])
        # Image.fromarray(origin_img).show()
        labels = cv2.imread(mask_list[i])
        # Image.fromarray(labels).show()
        # new_img = np.zeros((max_y-min_y, max_x-min_x, 3))
        new_mask_img = labels[min_y:max_y, min_x:max_x, :]

        # print(min_x, min_y, max_x, max_y)
        # cv2.rectangle(img, (min_x, min_y), (max_x, max_y), color=(0, 0, 255)) # 画标签框
        cv2.imwrite(mask_list[i].replace('Segmentation_labels_120X120', 'Segmentation_labels_no_background'), new_mask_img)
