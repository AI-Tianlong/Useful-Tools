import os
from PIL import Image
import numpy as np
from tqdm import trange

print('\n')
print(f'-----------第 1 步  转换训练集标签为18类---------------')

labels100_path = '/data/train_fusai/labels'

labels_list = os.listdir(labels100_path)

labels18_save_path = '/workspace/segmentation/Dataset/train_fusai/labels_18'

if not os.path.exists(labels18_save_path): os.mkdir(labels18_save_path)

class_list = [000, 101, 202, 303, 204, 205, 806, 807, 808,409, 410, 511, 512, 613, 614, 715, 716, 817]

print(f'---start label to 18')

for i in trange(len(labels_list),colour='GREEN',desc=f'---'):

    label = np.array(Image.open(os.path.join(labels100_path, labels_list[i])))

    for j in range(len(class_list)):

        label[np.where(label==class_list[j])] = j

    label = Image.fromarray(label)
    label.save(os.path.join(labels18_save_path, labels_list[i]))
print(f'---label to 18 have done')
print(f'---第 1 步 已完成')

print('\n')
print(f'-----------第 2 步  模型训练---------------')

