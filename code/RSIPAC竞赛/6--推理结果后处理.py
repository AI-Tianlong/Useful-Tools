from PIL import Image
import numpy as np
import mmcv
import os
import glob
from tqdm import trange
import time 

print('\n')
print(f'-----------第 6 步  results标签转换---------------')
print('\n')
time.sleep(5)

class_list = [000,101, 202, 303, 204, 205, 806, 807, 808,409, 410, 511, 512, 613, 614, 715, 716, 817]


label_path = '/workspace/segmentation/work_dirs/mask2former_beit_batch2_4w/results'
label_list = os.listdir(label_path)
label_save_path = label_path
if not os.path.exists(label_save_path):os.makedirs(label_save_path)

for index in trange(len(label_list)):
    label = np.array(Image.open(os.path.join(label_path,label_list[index])), dtype=np.int32)

    for i in range(len(class_list)):

        label[np.where(label==i)] = class_list[i]

    label = Image.fromarray(label)
    label.save(os.path.join(label_save_path,label_list[index]))

print(f'---第 6 步 已完成')

print('\n')
print(f'-----------第 7 步  压缩results.zip---------------')
print('\n')
