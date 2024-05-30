# Copyright (c) OpenMMLab. All rights reserved.
from osgeo import gdal
import argparse
import glob
import math
import os
import os.path as osp
from PIL import Image
import cv2
Image.MAX_IMAGE_PIXELS = None

import numpy as np
from mmengine.utils import ProgressBar, mkdir_or_exist
from tqdm import tqdm
from time import time
import logging
from ATL_Tools import mkdir_or_exist, find_data_list
 
# 配置日志输出格式
formatter = logging.Formatter('【ATL-Log】 %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
 
# 打印日志信息
# logger.info("这是一条日志信息")



def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert potsdam dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='dataset folder path')
    parser.add_argument('out_dir', help='output folder path')
    parser.add_argument(
        '--crop_size',
        type=int,
        help='clipped size of image after preparation',
        default=512)
    parser.add_argument(
        '--stride_size',
        type=int,
        help='stride of clipping original images',
        default=256)
    args = parser.parse_args()
    return args


def clip_big_image(image_path, save_path, splits, img_folder, label_folder, invalid_value, crop_size=512, youxiao_ratio=0.6):

    drop_ratio = 1 - youxiao_ratio

    img_basename = osp.basename(image_path) # S2_SR_2019_黑龙江省_七台河市_勃利县.tif
    
    data_type = 'train' if img_basename in splits['train'] else 'val'
    logger.info(f"当前处理的图像: {img_basename} ---> {data_type}")

    img_path_all_Big = image_path
    label_path_Big = image_path.replace(img_folder, label_folder)

    new_img_save_path = osp.join(save_path, 'img_dir', data_type)
    new_label_save_path = osp.join(save_path, 'ann_dir', data_type)

   
    # 读取图像
    image_gdal = gdal.Open(img_path_all_Big)
    img = image_gdal.ReadAsArray()
    img = img.transpose((1,2,0))
    img_h, img_w, img_bands = img.shape
    
    # 读取标签
    label_np = np.array(Image.open(label_path_Big))
    label_h, label_w = label_np.shape

    if img_h != label_h or img_w != label_w:
        logger.warning(f"图像和标签的大小不一致，图像大小为{img_h}x{img_w}，标签大小为{label_h}x{label_w}")
        if np.abs(img_h-label_h) < 10 and np.abs(img_w-label_w) < 10:
            label_np_resize = cv2.resize(label_np, (img_h, img_w), interpolation=cv2.INTER_NEAREST)
            label_np = label_np_resize
            logger.warning(f"经过resize后的，图像大小为{img_h}x{img_w}，标签大小为{label_h}x{label_w}")
        else:
            logger.warning(f"图像和标签的大小差距过大，不进行裁切")
            return
    # assert img_h == label_h and img_w == label_w, \
    #     f'图像和标签的大小不一致，图像大小为{img_h}x{img_w}，标签大小为{label_h}x{label_w}'
    
    # 裁切标签的512的部分
    for i in range(label_h//crop_size):
        for j in range(label_w//crop_size):
            new_label_512 = np.zeros((crop_size, crop_size), dtype=np.uint8)
            # 512小标签
            new_label_512 = label_np[i*crop_size:i*crop_size+crop_size, j*crop_size:j*crop_size+crop_size]     
            
            new_label_512_set = set(new_label_512.flatten()) # 检查里面有几种标签
            num_invalid = np.sum(new_label_512==0)           # 检查无效的像素个数
            total_pixels = new_label_512.size                # 总像素个数
            ratio_label_15 = num_invalid / total_pixels      # 15的像素占比
     

            # 根据标签判断是否保存当前标签图像: 如果无效标签的比例大于给定的阈值，则跳过不保存
            if ratio_label_15 <= drop_ratio:

                # 把标签中无效的地方变为255
                # new_label_512[new_label_512==15] = 255
                new_label_outpath = os.path.join(new_label_save_path, img_basename.split('.')[0]+'_'+str(i)+'_'+str(j)+'.tif')
                Image.fromarray(new_label_512).save(new_label_outpath)

                # 裁切图像的512的部分
                new_img_outpath = os.path.join(new_img_save_path, img_basename.split('.')[0]+'_'+str(i)+'_'+str(j)+'.tif') # 和label同名
                Driver = gdal.GetDriverByName("Gtiff")
                new_img_512 = np.zeros((crop_size,crop_size, img_bands), dtype=np.float32)
                # 512图像
                new_img_512_np = img[i*crop_size:i*crop_size+crop_size,j*crop_size:j*crop_size+crop_size,:] 
                new_img_gdal = Driver.Create(new_img_outpath, crop_size, crop_size, img_bands, image_gdal.GetRasterBand(1).DataType)    
                for band_num in range(img_bands):
                    band = new_img_gdal.GetRasterBand(band_num + 1)
                    band.WriteArray(new_img_512_np[:, :, band_num])
            else:
                pass


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    out_dir = args.out_dir
    mkdir_or_exist(out_dir)

    train_list = []
    val_list = []
    # 读取train_list.txt为列表
    with open(os.path.join(dataset_path, '黑龙江省-24类地物-train_list.txt'), 'r') as f:
        for line in f.readlines():
            train_list.append(line.strip())
    logger.info(f'train_list.txt包含 {len(train_list)} 张')

    # 读取val_list.txt为列表
    with open(os.path.join(dataset_path, '黑龙江省-24类地物-val_list.txt'), 'r') as f:
        for line in f.readlines():
            val_list.append(line.strip())
    logger.info(f'val_list.txt包含 {len(val_list)} 张') 

    splits = {
        'train': train_list,
        'val': val_list
    }

    logger.info('正在创建存储目录ing...')
    
    label_folder = os.path.join(dataset_path, '黑龙江省-24类地物-labels')
    img_folder = os.path.join(dataset_path, '黑龙江省-Sentinel2-images')

    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))
    
    # logger.info(f'Find the data {file_list}')

    image_lists = find_data_list(img_folder, '.tif')
    for image_path in tqdm(image_lists):
        tqdm.desc = f"---正在切分图像{os.path.basename(image_path)}"
        tqdm.colour = "GREEN"
        clip_big_image(image_path, out_dir, splits, img_folder, label_folder, crop_size=512, youxiao_ratio=0.6, invalid_value=0)
        # prog_bar.update()

    logger.info('Done!')
    
    num_image_val = find_data_list(osp.join(out_dir, 'img_dir', 'val'), '.tif')
    num_image_train = find_data_list(osp.join(out_dir, 'img_dir', 'train'), '.tif')
    num_label_val = find_data_list(osp.join(out_dir, 'ann_dir', 'val'), '.tif')
    num_label_train = find_data_list(osp.join(out_dir, 'ann_dir', 'train'), '.tif')

    logger.info(f'训练集图像数量: {len(num_image_train)}')
    logger.info(f'验证集图像数量: {len(num_image_val)}')  
    logger.info(f'训练集标签数量: {len(num_label_train)}')
    logger.info(f'验证集标签数量: {len(num_label_val)}')

if __name__ == '__main__':
    main()
