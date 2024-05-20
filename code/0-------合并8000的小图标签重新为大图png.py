from PIL import Image
import os
import numpy as np
from ATL_Tools import mkdir_or_exist, find_data_list

from osgeo import gdal
from tqdm import tqdm

# 思路-->合并之后再去转换RGB，还是转完RGB再去合并？？？

# 合并之后再去转RGB吧，这样，还能去合并类别。把几个类别合并成新的类别，因为很乱，然后再转RGB。

from PIL import Image
import os
import numpy as np
from ATL_Tools import mkdir_or_exist, find_data_list

from osgeo import gdal
from tqdm import tqdm

# 思路-->合并之后再去转换RGB，还是转完RGB再去合并？？？

# 合并之后再去转RGB吧，这样，还能去合并类别。把几个类别合并成新的类别，因为很乱，然后再转RGB。


def main():

    crop_size = 5000

    # 小图标签的存储位置
    img_patch_path = '/opt/AI-Tianlong/Datasets/ATL-ATLNongYe/ATL推理大图/双鸭山/5000的小图/小图_24类_mask'
    img_patch_list = find_data_list(img_patch_path, suffix='.png')

    # 合并后的大图的存储位置：
    img_big_path = '/opt/AI-Tianlong/Datasets/ATL-ATLNongYe/ATL推理大图/双鸭山/5000的小图/大图_24类_mask'
    mkdir_or_exist(img_big_path)

    # 原始的大图的存储位置：为了读取尺寸
    ori_img_big_path = '/opt/AI-Tianlong/Datasets/ATL-ATLNongYe/ATL推理大图/双鸭山/5000的小图/大图'
    ori_img_big_list = find_data_list(ori_img_big_path, suffix='.tif')

    # 这里需要知道每一个大图的尺寸，才能去合并。
    # 所以还需要先打开大图，读出里面的shape
    for ori_img_path in tqdm(ori_img_big_list, colour='Green'):
        # 读取最原始的大图，获取尺寸
        # 南岗区.tif
        img_basename = os.path.basename(ori_img_path)
        print(f"正在处理: {img_basename}")

        img_ori = gdal.Open(ori_img_path)

        h = img_ori.RasterYSize
        w = img_ori.RasterXSize
        c = img_ori.RasterCount
        
        print(f'当前大图的尺寸为：({h}, {w}, {c})')
        
        # 创建一个新的大图，用来存储合并后的结果
        new_big_img = np.zeros((h,w),dtype=np.uint8)
        # "南岗区"
        img_basename_no_suffix = img_basename.split('.')[0]

        for img_patch_path in img_patch_list:
            if img_basename_no_suffix in img_patch_path:
               # 'S2_SR_2019_黑龙江省_双鸭山市_宝清县' in 'S2_SR_2019_黑龙江省_双鸭山市_宝清县_0_0_5000.png' 
                
                img_patch = np.array(Image.open(img_patch_path))
          
                # 处理左上的
                if (not 'xia' in img_patch_path) and (not 'you' in img_patch_path) and (not 'youxia' in img_patch_path):
                    img_h = int(os.path.basename(img_patch_path).split('_')[-3]) # [1]
                    img_w = int(os.path.basename(img_patch_path).split('_')[-2]) # [2]
                    new_big_img[img_h*crop_size:img_h*crop_size+crop_size, img_w*crop_size:img_w*crop_size+crop_size] = img_patch
                
                # 处理右的
                elif (not 'youxia' in img_patch_path) and ('xia' in img_patch_path):
                    img_w = int(os.path.basename(img_patch_path).split('_')[-2]) # [2]
                    new_big_img[h-crop_size:h, img_w*crop_size:img_w*crop_size+crop_size] = img_patch

                # 处理下的
                elif (not 'youxia' in img_patch_path) and ('you' in img_patch_path):
                    img_h = int(os.path.basename(img_patch_path).split('_')[-2]) # [2]
                    new_big_img[img_h*crop_size:img_h*crop_size+crop_size,w-crop_size:w] = img_patch

                elif 'youxia' in img_patch_path:
                    new_big_img[h-crop_size:h,w-crop_size:w] = img_patch

        Image.fromarray(new_big_img).save(os.path.join(img_big_path, img_basename_no_suffix + '.png'))

if __name__ == '__main__':
    main()

    # 合并后的大图的存储位置：
    img_big_path = '/opt/AI-Tianlong/Datasets/ATL-ATLNongYe/ATL推理大图/双鸭山/5000的小图/大图_24类_mask'
    img_big_lists = find_data_list(img_big_path, suffix='.tif')
    print(f"已经将小图合并成 {len(img_big_lists)} 张大图")
def main():

    crop_size = 8000

    # 小图标签的存储位置
    img_patch_path = '/opt/AI-Tianlong/Datasets/ATL_DATASETS/Harbin/Harbin_8000_labels'
    img_patch_list = find_data_list(img_patch_path, suffix='.png')

    # 合并后的大图的存储位置：
    img_big_path = '/opt/AI-Tianlong/Datasets/ATL_DATASETS/Harbin/Harbin_big_labels'
    mkdir_or_exist(img_big_path)


    # 原始的大图的存储位置：为了读取尺寸
    ori_img_big_path = '/opt/AI-Tianlong/Datasets/ATL_DATASETS/Harbin/images_3channel'
    ori_img_big_list = find_data_list(ori_img_big_path, suffix='.tif')

    # 这里需要知道每一个大图的尺寸，才能去合并。
    # 所以还需要先打开大图，读出里面的shape
    for ori_img_path in tqdm(ori_img_big_list, colour='Green'):
        # 读取最原始的大图，获取尺寸
        # 南岗区.tif
        img_basename = os.path.basename(ori_img_path)
        print(f"正在处理: {img_basename}")

        img_ori = gdal.Open(ori_img_path)
        img_ori = img_ori.ReadAsArray()
        img_ori = img_ori.transpose((1,2,0))
        h, w, c = img_ori.shape
        
        # 创建一个新的大图，用来存储合并后的结果
        new_big_img = np.zeros((h,w),dtype=np.uint8)
        # "南岗区"
        img_basename_no_suffix = img_basename.split('.')[0]

        for img_patch_path in img_patch_list:
            if img_basename_no_suffix in img_patch_path:
                
                h_index = 1
                w_index = 2

                if '10m' in img_patch_path:
                    h_index = 2
                    w_index = 3

                
                img_patch = np.array(Image.open(img_patch_path))
                # 如果南岗区.tif = 南岗区.tif,则跳过
                if os.path.basename(img_patch_path).split('.')[0] == img_basename_no_suffix:
                    new_big_img = img_patch               
                # 处理左上的
                elif (not 'xia' in img_patch_path) and (not 'you' in img_patch_path) and (not 'youxia' in img_patch_path):
                    img_h = int(os.path.basename(img_patch_path).split('_')[h_index]) # [1]
                    img_w = int(os.path.basename(img_patch_path).split('_')[w_index]) # [2]
                    new_big_img[img_h*crop_size:img_h*crop_size+crop_size, img_w*crop_size:img_w*crop_size+crop_size] = img_patch
                
                # 处理右的
                elif (not 'youxia' in img_patch_path) and ('xia' in img_patch_path):
                    img_w = int(os.path.basename(img_patch_path).split('_')[w_index]) # [2]
                    new_big_img[h-crop_size:h, img_w*crop_size:img_w*crop_size+crop_size] = img_patch

                # 处理下的
                elif (not 'youxia' in img_patch_path) and ('you' in img_patch_path):
                    img_h = int(os.path.basename(img_patch_path).split('_')[w_index]) # [2]
                    new_big_img[img_h*crop_size:img_h*crop_size+crop_size,w-crop_size:w] = img_patch

                elif 'youxia' in img_patch_path:
                    new_big_img[h-crop_size:h,w-crop_size:w] = img_patch

        Image.fromarray(new_big_img).save(os.path.join(img_big_path, img_basename))

if __name__ == '__main__':
    main()

    # 合并后的大图的存储位置：
    img_big_path = '/opt/AI-Tianlong/Datasets/ATL_DATASETS/Harbin/Harbin_big_labels'
    img_big_lists = find_data_list(img_big_path, suffix='.tif')
    print(f"已经将小图合并成 {len(img_big_lists)} 张大图")
