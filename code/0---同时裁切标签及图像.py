# Copyright (c) OpenMMLab. All rights reserved.
from osgeo import gdal
import argparse
import glob
import math
import os
import os.path as osp
from PIL import Image
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
    parser.add_argument('dataset_path', help='potsdam folder path')
    # parser.add_argument('--bit', help='potsdam folder path', default='8bit',)
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--clip_size',
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


def clip_big_image(image_path, save_path, splits, img_folder, label_folder, crop_size=512, ):
    # Original image of ｛ATL Sentinel-2｝ dataset is very large, 
    # thus pre-processing of them is adopted. 
    # Given fixed clip size and stride size to generate
    # clipped image, the intersection　of width and height is determined.
    # For example, given one 5120 x 5120 original image, the clip size is
    # 512 and stride size is 256, thus it would generate 20x20 = 400 images
    # whose size are all 512x512.

    img_basename = osp.basename(image_path)# S2_SR_2019_黑龙江省_七台河市_勃利县.tif
    data_type = 'train' if img_basename in splits['train'] else 'val'
    
    logger.info(f"当前处理的图像的base_name: {img_basename} ---> {data_type}")

    img_path_all = image_path
    label_path_all = image_path.replace('images-xian','labels-xian-new')

    new_img_save_path = osp.join(save_path, 'img_dir', data_type)
    new_label_save_path = osp.join(save_path, 'ann_dir', data_type)

    # 同时裁切图像和标签，如果里面有label==15，则跳过不保存
    
    # 读取图像
    image_gdal = gdal.Open(img_path_all)
    img = image_gdal.ReadAsArray()
    img = img.transpose((1,2,0))
    img_h, img_w, img_bands = img.shape
    
    # 读取标签
    label_np = np.array(Image.open(label_path_all))
    label_h, label_w = label_np.shape

    assert img_h == label_h and img_w == label_w, \
        f'图像和标签的大小不一致，图像大小为{img_h}x{img_w}，标签大小为{label_h}x{label_w}'
    
    # 裁切标签的512的部分
    for i in range(label_h//crop_size):
        for j in range(label_w//crop_size):
            new_label_512 = np.zeros((crop_size,crop_size),dtype=np.uint8)
            new_label_512 = label_np[i*crop_size:i*crop_size+crop_size,j*crop_size:j*crop_size+crop_size]   #横着来       
            
            # 根据标签判断是否保存当前标签图像: 如果里面有label==15，则跳过不保存
            if not 15 in new_label_512:           # ann_dir/train/S2_SR_2019_黑龙江省_七台河市_勃利县_0_0.tif
                new_label_outpath = os.path.join(new_label_save_path, img_basename.split('.')[0]+'_'+str(i)+'_'+str(j)+'.tif')
                Image.fromarray(new_label_512).save(new_label_outpath)

                # 裁切图像的512的部分
                new_img_outpath = os.path.join(new_img_save_path, img_basename.split('.')[0]+'_'+str(i)+'_'+str(j)+'.tif')
                Driver = gdal.GetDriverByName("Gtiff")
                new_img_512 = np.zeros((crop_size,crop_size, img_bands),dtype=np.float32)
                new_img_512 = img[i*crop_size:i*crop_size+crop_size,j*crop_size:j*crop_size+crop_size,:]   #横着来   
                new_img_gdal = Driver.Create(new_img_outpath, crop_size, crop_size, img_bands, gdal.GDT_Float32)    
                for band_num in range(img_bands):
                    band = new_img_gdal.GetRasterBand(band_num + 1)
                    band.WriteArray(new_img_512[:, :, band_num])
            else:
                pass


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = os.path.join(args.dataset_path, 'data', 'atl_s2_crop_10m')
    else:   
        out_dir = args.out_dir
    
    mkdir_or_exist(out_dir)

    train_list = []
    val_list = []
    # 读取train_list.txt为列表
    with open(os.path.join(dataset_path, 'train_list.txt'), 'r') as f:
        for line in f.readlines():
            train_list.append(line.strip())
    logger.info(f'train_list.txt包含 {len(train_list)} 张')

    # 读取val_list.txt为列表
    with open(os.path.join(dataset_path, 'val_list.txt'), 'r') as f:
        for line in f.readlines():
            val_list.append(line.strip())
    logger.info(f'val_list.txt包含 {len(val_list)} 张')

    splits = {
        'train': train_list,
        'val': val_list
    }

    logger.info('Making directories...')
    
    label_folder = os.path.join(dataset_path, 'labels-xian-new')
    img_folder = os.path.join(dataset_path, 'images-xian')

    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))
    
    # logger.info(f'Find the data {file_list}')
    image_lists = find_data_list(img_folder, '.tif')

    for image_path in tqdm(image_lists):
        tqdm.desc = f"---正在切分图像{os.path.basename(image_path)}"
        tqdm.colour = "GREEN"
        clip_big_image(image_path, out_dir, splits, img_folder, label_folder, crop_size=512)
        # prog_bar.update()

    logger.info('Done!')


if __name__ == '__main__':
    main()
