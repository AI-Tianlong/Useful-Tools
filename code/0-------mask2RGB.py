import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from ATL_Tools import mkdir_or_exist, find_data_list
import os

MASK_path = './dataset/potsdam/mask_labels/'
RGB_path = 'dataset/potsdam/vis_labels'
mkdir_or_exist(RGB_path)

label_lists = find_data_list(MASK_path, suffix='png')

# 思路：三个通道分别乘 2 3 4，然后相加，得到一个新的通道，然后根据这个通道的值，来判断是哪个类别
classes = ['Background', 'Building', 'di_ai_zhi_bei',
         'Tree', 'car', 'bu_tou_shui_mian']

platte = [[255, 0, 0], [0, 0, 255], [0, 255, 255],
          [0, 255, 0], [255, 255, 0], [255, 255, 255]]

platte = np.array(platte)

for mask_label_path in tqdm(label_lists):
    mask_label = np.array(Image.open(mask_label_path))
    RGB_label = platte[mask_label]
    Image.fromarray(RGB_label.astype(np.uint8)).save(os.path.join(RGB_path, os.path.basename(mask_label_path).replace('.png', '.png')))
