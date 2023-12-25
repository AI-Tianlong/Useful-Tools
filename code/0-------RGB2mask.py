import numpy as np
from PIL import Image
from tqdm import tqdm, trange
# 第一次使用，请先 `pip install ATL_Tools`
from ATL_Tools import mkdir_or_exist, find_data_list
import os

RGB_path = 'dataset/potsdam/RGB_labels'
MASK_path = 'dataset/potsdam/mask_labels/'
mkdir_or_exist(MASK_path)

RGB_label_lists = find_data_list(RGB_path, suffix='.tif')

# 思路：三个通道分别乘 2 3 4，然后相加，得到一个新的通道，然后根据这个通道的值，来判断是哪个类别
classes = ['cluster', 'Building', 'di_ai_zhi_bei',
         'Tree', 'car', 'bu_tou_shui_mian']

platte = [[255, 0, 0], [0, 0, 255], [0, 255, 255],
          [0, 255, 0], [255, 255, 0], [255, 255, 255]]

# 使用这样一个platte和classes的好处就是和Openmmlab的各个库的颜色对应上了，这样就可以直接用他们的代码了
# 这个列表里索引 `2R+3G+4B` 的位置，存的是类别的值，
# 即，用RGB算出来一个值XXX，platte_idx[XXX]的值，就是这个像素的类别
# 2R+3G+4B --> 
platte_idx = np.zeros(256*10)
for i, platte_i in enumerate(platte):
    platte_idx[np.matmul(platte_i, np.array([2,3,4]).reshape(3,1))] = i 
    
for RGB_label_path in tqdm(RGB_label_lists):
    RGB_label = np.array(Image.open(RGB_label_path)).astype(np.uint8)  # 6000*6000*3
    #把一张图展开，6000*6000*3 --> 36000000*3
    #然后36000000*3 * 3*1 = 36000000*1 
    #然后，去platte_idx里找到对应的类别
    RGB_idx =  np.matmul(RGB_label.reshape(-1, RGB_label.shape[2]), np.array([2,3,4]).reshape(3,1))
    # 调试用
    # print(f'RGB_idx里包含的数值：{set(RGB_idx.reshape(-1,))} 对应的类别：{set(platte_idx[RGB_idx].reshape(-1,))}')
    mask_label = platte_idx[RGB_idx].astype(np.uint8)# 6000*6000
    Image.fromarray(mask_label.reshape(RGB_label.shape[0], RGB_label.shape[1])).save(
        os.path.join(MASK_path, os.path.basename(RGB_label_path).replace('.tif', '.png')))

        
