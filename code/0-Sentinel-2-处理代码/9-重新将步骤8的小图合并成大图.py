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
    img_patch_path = '/opt/AI-Tianlong/Datasets/ATL-ATLNongYe/ATL推理大图/牡丹江/推理结果-3类/推理结果-3-mask-小图'
    img_patch_list = find_data_list(img_patch_path, suffix='.png')

    # 合并后的大图的存储位置：
    img_big_path = '/opt/AI-Tianlong/Datasets/ATL-ATLNongYe/ATL推理大图/牡丹江/推理结果-3类/推理结果-3-mask'
    mkdir_or_exist(img_big_path)

    # 原始的大图的存储位置：为了读取尺寸
    ori_img_big_path = '/opt/AI-Tianlong/Datasets/ATL-ATLNongYe/ATL推理大图/牡丹江/要推理的images-矢量裁切'
    ori_img_big_list = find_data_list(ori_img_big_path, suffix='.tif')

    # 这里需要知道每一个大图的尺寸，才能去合并。
    # 所以还需要先打开大图，读出里面的shape
    for ori_img_path in tqdm(ori_img_big_list, colour='Green'):
        # 读取最原始的大图，获取尺寸
        # 南岗区.tif
        img_basename = os.path.basename(ori_img_path)
        print(f'==============================================')
        print(f'【ATL-LOG】当前要合并的大图为：{img_basename}')

        img_ori = gdal.Open(ori_img_path)

        h = img_ori.RasterYSize
        w = img_ori.RasterXSize
        c = img_ori.RasterCount

        print(f'【ATL-LOG】当前大图的尺寸为：({h}, {w}, {c})')

        
        # 创建一个新的大图，用来存储合并后的结果
        new_big_img = np.zeros((h,w),dtype=np.uint8)
        # "南岗区"
        img_basename_no_suffix = img_basename.split('.')[0]

        # 查找小图文件夹中,包含大图名字的路径
        path_img_merge_list = []
        for img_patch_path in img_patch_list:
            if img_basename_no_suffix in img_patch_path:
                path_img_merge_list.append(img_patch_path)
        print(f'符合{img_basename_no_suffix}的有 {len(path_img_merge_list)} 张')

        for patch_path in path_img_merge_list:
            if os.path.basename(patch_path).split('.')[0] == img_basename_no_suffix:
                print(f'小图没有序号后缀, {img_basename} 有边长小于 {crop_size}, 未进行裁切, 不需要合并')
                new_big_img = np.array(Image.open(patch_path))
            else: 
                img_patch = np.array(Image.open(patch_path))
            
                # 处理左上的
                if (not 'xia' in patch_path) and (not 'you' in patch_path) and (not 'youxia' in patch_path):
                    img_h = int(os.path.basename(patch_path).split('_')[-3]) # [1]
                    img_w = int(os.path.basename(patch_path).split('_')[-2]) # [2]
                    new_big_img[img_h*crop_size:img_h*crop_size+crop_size, img_w*crop_size:img_w*crop_size+crop_size] = img_patch
                
                # 处理右的
                elif (not 'youxia' in patch_path) and ('xia' in patch_path):
                    img_w = int(os.path.basename(patch_path).split('_')[-2]) # [2]
                    new_big_img[h-crop_size:h, img_w*crop_size:img_w*crop_size+crop_size] = img_patch

                # 处理下的
                elif (not 'youxia' in patch_path) and ('you' in patch_path):
                    img_h = int(os.path.basename(patch_path).split('_')[-2]) # [2]
                    new_big_img[img_h*crop_size:img_h*crop_size+crop_size,w-crop_size:w] = img_patch

                elif 'youxia' in patch_path:
                    new_big_img[h-crop_size:h,w-crop_size:w] = img_patch

        Image.fromarray(new_big_img).save(os.path.join(img_big_path, img_basename_no_suffix + '.png'))

    print(f"已经将小图合并成 {len(find_data_list(img_big_path, suffix='.png'))} 张大图")

if __name__ == '__main__':
    main()
