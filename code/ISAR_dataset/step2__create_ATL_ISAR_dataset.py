import os 
import numpy as np
from path import scandir, find_data_list, mkdir_or_exist
from tqdm import trange, tqdm

from typing import List
from PIL import Image

# =====================================需要修改的一些参数===================================================

img_root_path = '../Dataset_test/SynISAR_no_background'   # 原始数据集的根目录
new_img_save_root_path = '../Dataset_test/ATL_ISAR'        # 要生成数据集的根目录
mkdir_or_exist(new_img_save_root_path)

num_train_images = 30    # 要生成的数据集的张数
num_val_images = 10    # 要生成的数据集的张数

new_img_size = [1024, 1024, 3]  # 要生成的图像的尺寸
original_img_size = [120, 120, 3] # 原始数据集的大小


img_list = find_data_list(img_root_path, suffix='jpg') # 原始数据集的路劲列表



def create_ISAR_dateset(num_images = 2000, type = 'train', cover = False):
    """生成带有标签的检测数据集
    
    Args:
        num_images (uint): 要生成的数据集的数量
        type (str): 'train' or 'val', 选择要生成训练集还是验证集,和路径有关系
        cover (bool): Default `False`, 是否要生成边缘处有裁切的数据集
    
    """

    mkdir_or_exist(os.path.join(new_img_save_root_path, type,'images'))
    mkdir_or_exist(os.path.join(new_img_save_root_path, type,'labels'))

    for i in trange(num_images):
        num_plane = np.random.randint(1, 10)  # 生成的图中的飞机的个数
        index_list = np.random.randint(0, len(img_list), size=num_plane) # 从数据集中选取的图像索引列表
        # new_big_img_size = np.array(new_img_size) + 2*np.array([original_img_size[0], original_img_size[1], 0])
        new_img = np.zeros(shape=new_img_size, dtype=np.uint8) # 要生成大大大图像素的初始化
        new_img[:, :, 2] = 126 # 要生成图像的底色和飞机的底色弄得一致
        # print(new_big_img_size)
        # Image.fromarray(new_big_img).show()
        # print(f'-- 生成图像中有飞机目标数：{num_plane}')

        # ------------------------------------- 嵌入图片 + 生成标签---------------------------------------
        with open(os.path.join(new_img_save_root_path, type, 'labels', str(i)+'.csv'), 'w+') as f:
            f.write('filename, x1, y1, x2, y2, class\n')
            for j in index_list:    
                img = np.array(Image.open(img_list[j]))  # 把要嵌入的图片读进来
                img_height, img_width = img.shape[0], img.shape[1] # 要嵌入图片的高和宽
                background = np.zeros(img.shape,dtype=np.uint8)
                background[:,:,2] = 126
                # Image.fromarray(background).show()
                # 在new_img_size+120的大图上随机选一个起始点，然后把图像的左上角对齐这个点，嵌入进去，
                # 再从大图上裁出一个1024*1024的，就避免了边宽分类的这个问题

                # if cover:  # 边缘部分有裁切
                #     new_h, new_w = np.random.randint(0, new_big_img_size[0]-original_img_size[0]), \
                #         np.random.randint(0, new_img_size[1]-original_img_size[1])
                # else:
                new_h, new_w = np.random.randint(original_img_size[0], new_img_size[0]-original_img_size[0]), \
                    np.random.randint(original_img_size[1], new_img_size[1]-original_img_size[1])

                if (new_img[new_h:new_h+img_height, new_w:new_w+img_width, :] == background).all():  # 保证不重叠把图嵌上去
                    
                    # -----------嵌入图片
                    new_img[new_h:new_h+img_height, new_w:new_w+img_width, :] = img

                    # -----------生成标签               
                    if 'Aircraft-1' in img_list[j] or 'Aircraft-2' in img_list[j] \
                        or 'Aircraft-3' in img_list[j] \
                        or 'Aircraft-4' in img_list[j] or 'Aircraft-7'in img_list[j]:
                        class_name = 'ZhanDouJi'
                    elif 'Aircraft-5' in img_list[j]:
                        class_name = 'YunShuJi'
                    elif 'Aircraft-6' in img_list[j]:
                        class_name = 'BaJi'
                    
                    # 在1024*1024图像上的图像的坐标，涉及到一个变化
                    x1, y1, x2, y2 = new_w, new_h, new_w+img_width, new_h+img_height

                    csv_label = img_list[j] + ',' + str(x1) + ',' +str(y1) + ',' +str(x2) + ',' +str(y2) + ',' + class_name + '\n'
                    f.write(csv_label)
                    f.flush()
                
                else:
                    pass # 跳过这一张图
                
        # 嵌入完的最后，裁切一个new_img_size的出来
        # new_img = new_img[original_img_size[0]:original_img_size[0]+new_img_size[0], \
        #                     original_img_size[1]:original_img_size[1]+new_img_size[1], :]

        Image.fromarray(new_img).save(os.path.join(new_img_save_root_path, type, 'images', str(i)+'.png'))

    print(f'-- 创建 {type} 数据集完成，共创建图像 {num_images} 张，位于{os.path.join(new_img_save_root_path, type)} \n')
    with open(os.path.join(new_img_save_root_path, type, 'classes.txt'), 'w+') as f:
        f.write('1 ZhanDouJi\n')
        f.write('2 YunShuJi\n')
        f.write('3 BaJi\n')
        f.flush()

def main():
    create_ISAR_dateset(num_images=num_train_images, type='train')
    create_ISAR_dateset(num_images=num_val_images, type='val')
    print(print('==============================================================\n'))

if __name__ == '__main__':
    main()