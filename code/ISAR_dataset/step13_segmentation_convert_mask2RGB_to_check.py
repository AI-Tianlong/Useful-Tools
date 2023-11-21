# 本程序使用来裁切图像的，找到有目标的边框，并直接裁切

import cv2
import numpy as np
import os
from path import mkdir_or_exist, find_data_list
from PIL import Image
from tqdm import trange

# 根据img图像去检测bbox, 然后把同样的bbox 作用在mask上，应该可 

print('\n====================== Step 12 Convert RGB label to mask ======================\n')

# 数据集的地址    
RGB_label_root_path = '../Dataset_test/ATL_ISAR_Seg_dataset'
mask_list = find_data_list(RGB_label_root_path, '.png')




COLORMAP = dict(
    Background=(0, 0, 0),   # 0-背景-黑色
    Empennage=(128, 0, 0),  # 1-尾翼-红色
    Engine=(0, 128, 0),     # 2-引擎-绿色
    Fuselage=(128, 128, 0), # 3-机身-黄色
    Head=(0, 0, 128),       # 4-机头-蓝色
    Wing=(128, 0, 128),     # 5-机翼-紫色
)
palette = list(COLORMAP.values())
classes = list(COLORMAP.keys())

def mask2RGB(mask_path, vis_path):
    new_mask = np.array(Image.open(mask_path)).astype(np.uint8) #把原来的图片读出来，存成uint8的格式
    cm = np.array(list(COLORMAP.values())).astype(np.uint8) #取出COLOR_MAP中的值，存到列表
    color_img = cm[new_mask]  #这里就变成1024*1024*3的了？ why？？
    color_img = Image.fromarray(np.uint8(color_img)) #从np的array格式转换成PIL格式
    color_img.save(vis_path)

#############用列表来存一个 RGB 和一个类别的对应################
def colormap2label(palette):   
    colormap2label_list = np.zeros(256**3, dtype = np.longlong)
    for i, colormap in enumerate(palette):
        colormap2label_list[(colormap[0] * 256 + colormap[1])*256+colormap[2]] = i
    return colormap2label_list

#############给定那个列表，和vis_png然后生成masks_png################
def label_indices(RGB_label, colormap2label_list):
    RGB_label = RGB_label.astype('int32')
    idx = (RGB_label[:, :, 0] * 256 + RGB_label[:, :, 1]) * 256 + RGB_label[:, :, 2]
    # print(idx.shape)
    return colormap2label_list[idx]

def RGB2mask(RGB_path, mask_path, colormap2label_list):
    RGB_label = np.array(Image.open(RGB_path).convert('RGB')) #打开RGB_png
    mask_label = label_indices(RGB_label, colormap2label_list) # .numpy()
    mask_label = Image.fromarray(np.uint8(mask_label)) #从np的array格式转换成PIL格式
    mask_label.save(mask_path)

def render(mask_path, vis_path):
    new_mask = np.array(Image.open(mask_path)).astype(np.uint8) #把原来的图片读出来，存成uint8的格式
    cm = np.array(palette).astype(np.uint8) #取出COLOR_MAP中的值，存到列表
    color_img = cm[new_mask]  #这里就变成1024*1024*3的了？ why？？
    color_img = Image.fromarray(np.uint8(color_img)) #从np的array格式转换成PIL格式
    color_img.save(vis_path)


if __name__ == '__main__':
    print('vis_png 转换ing...')

    colormap2label_list = colormap2label(palette)

    for i in trange(len(mask_list)):

        render(os.path.join(mask_list[i]), os.path.join(mask_list[i])) #这里得改一下

    print(f'共转换 RGB label to mask {len(mask_list)} 张')