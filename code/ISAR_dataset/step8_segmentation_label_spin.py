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



def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    # bound_w = int(height * abs_sin + width * abs_cos)
    # bound_h = int(height * abs_cos + width * abs_sin)

    bound_w = 120
    bound_h = 120

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h),flags=cv2.INTER_NEAREST)
    
    return rotated_mat



print('\n====================== Step 8 Create Segmentation labels =======================\n')


img_root_dir = '../Dataset_test/SynISAR_low_resolution'
labels_root_dir = '../Dataset_test/Segmentation_labels_120X120'

for j in range(1, 7):
    
    labels_root_dir_temp = os.path.join(labels_root_dir, 'Aircraft-'+str(j))
    print(f'labels_root_dir_temp={labels_root_dir_temp}')
    # img_root_dir_temp = os.path.join(img_root_dir, 'Aircraft-' + str(j))
    # mkdir_or_exist(labels_root_dir_temp)

    # img_temp = cv2.imread(os.path.join(img_root_dir_temp, '000.png'))
    # cv2.imwrite(os.path.join(labels_root_dir_temp, '000.png'), img_temp)
    # print(f'Creating {labels_root_dir_temp}')

    for i in trange(180):
        label = cv2.imread(os.path.join(labels_root_dir_temp, '000.png'))

        new_label = rotate_image(label, i*1)
        cv2.imwrite(os.path.join(labels_root_dir_temp, str(i+1).zfill(3)+'.png') ,new_label)


labels_root_dir_temp = os.path.join(labels_root_dir, 'Aircraft-'+str(7))
# img_root_dir_temp = img_root_dir + '\Aircraft-' + str(7)
mkdir_or_exist(labels_root_dir_temp)
# img_temp = cv2.imread(os.path.join(img_root_dir_temp, '000.png'))
# cv2.imwrite(os.path.join(labels_root_dir_temp, '000.png'), img_temp)
print(f'Creating {labels_root_dir_temp}')

for i in trange(180):
    label = cv2.imread(os.path.join(labels_root_dir_temp, '000.png'))

    new_label = rotate_image(label, i*1.9)
    cv2.imwrite(os.path.join(labels_root_dir_temp, str(i+1).zfill(3)+'.png') ,new_label)

print('\n===================================== Done =====================================\n')


