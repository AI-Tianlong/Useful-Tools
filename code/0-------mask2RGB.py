import numpy as np
from PIL import Image
from tqdm import tqdm, trange
# 第一次使用 请`pip install ATL_Tools`
from ATL_Tools import mkdir_or_exist, find_data_list
import os

MASK_path = '/opt/AI-Tianlong/Datasets/LoveDA/ann_dir/val'
RGB_path = '/opt/AI-Tianlong/Datasets/LoveDA/ann_dir/vis_val'
mkdir_or_exist(RGB_path)

label_lists = find_data_list(MASK_path, suffix='png')

# 思路：三个通道分别乘 2 3 4，然后相加，得到一个新的通道，然后根据这个通道的值，来判断是哪个类别

# 是否包含0类，classes和palette里没包含
reduce_zero_label = True

classes = ('background', 'building', 'road', 'water', 'barren', 'forest',
            'agricultural')

palette = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
            [159, 129, 183], [0, 255, 0], [255, 195, 128]]

# palette = np.array(palette)

if reduce_zero_label:
    new_palette = [[0, 0, 0]] + palette
    print(f"palette: {new_palette}")
else:
    new_palette = palette
    print(f"palette: {new_palette}")

new_palette = np.array(new_palette)
for mask_label_path in tqdm(label_lists):
    mask_label = np.array(Image.open(mask_label_path)).astype(np.uint8)
    # print(mask_label.shape)
    RGB_label = new_palette[mask_label]
    Image.fromarray(RGB_label.astype(np.uint8)).save(os.path.join(RGB_path, os.path.basename(mask_label_path).replace('.png', '.png')))
