import os
import shutil 
import numpy as np
from path import scandir, find_data_list, mkdir_or_exist
from tqdm import trange, tqdm
import cv2
from typing import List
from PIL import Image
from pandas import read_csv


# =====================================需要修改的一些参数===================================================
img_ATL_ISAR_path = '../Dataset_test/ATL_ISAR'

img_Add_Noise_path = '../Dataset_test/ATL_Add_Noise'   # 原始数据集的根目录


def add_noise(img_Add_Noise_path, dataset_type = 'train'):

    mkdir_or_exist(os.path.join(img_Add_Noise_path, dataset_type, 'images'))
    # mkdir_or_exist(os.path.join(img_Add_Noise_path, dataset_type, 'labels'))

    img_list = os.listdir(os.path.join(img_ATL_ISAR_path, dataset_type,'images'))
    label_list = os.listdir(os.path.join(img_ATL_ISAR_path, dataset_type,'labels'))

    for i in trange(len(img_list), desc=f'正在添加噪声 {dataset_type} ', colour='GREEN'):
        image_whole_path = os.path.join(img_ATL_ISAR_path, dataset_type, 'images', img_list[i])
        # print(f'image_whole_path={image_whole_path}')
        # 读取图像文件
        origin_img = cv2.imread(image_whole_path)

        # -----------------高斯噪声----------
        mean = 0
        #设置高斯分布的标准差
        sigma = 25
        #根据均值和标准差生成符合高斯分布的噪声
        gauss = np.random.normal(mean,sigma, origin_img.shape)
        #给图片添加高斯噪声
        noisy_img = origin_img + gauss
        #设置图片添加高斯噪声之后的像素值的范围
        noisy_img = np.clip(noisy_img,a_min=0,a_max=255)

        # -----------------泊松噪声----------
        vals = len(np.unique(noisy_img))
        vals = 2 ** np.ceil(np.log2(vals))
        #给图片添加泊松噪声
        noisy_img = np.random.poisson(noisy_img * vals) / float(vals)

        # -----------------椒盐噪声---------
        s_vs_p = 0.5
        #设置添加噪声图像像素的数目
        amount = 0.04
        noisy_img = np.copy(noisy_img)
        #添加salt噪声
        num_salt = np.ceil(amount * noisy_img.size * s_vs_p)
        #设置添加噪声的坐标位置
        coords = [np.random.randint(0,i - 1, int(num_salt)) for i in noisy_img.shape]
        noisy_img[coords[0],coords[1],:] = [255,255,255]
        #添加pepper噪声
        num_pepper = np.ceil(amount * noisy_img.size * (1. - s_vs_p))
        #设置添加噪声的坐标位置
        coords = [np.random.randint(0,i - 1, int(num_pepper)) for i in noisy_img.shape]
        noisy_img[coords[0],coords[1],:] = [0,0,0]

        # --------------speckle噪声-------------
        gauss = np.random.randn(1024,1024,3)
        #给图片添加speckle噪声
        noisy_img = noisy_img + noisy_img * gauss
        #归一化图像的像素值
        noisy_img = np.clip(noisy_img,a_min=0,a_max=255)

        cv2.imwrite(os.path.join(img_Add_Noise_path, dataset_type, 'images', img_list[i]), noisy_img)


if __name__ == '__main__':
    add_noise(img_Add_Noise_path, dataset_type='train')
    add_noise(img_Add_Noise_path, dataset_type='val')

    # 然后再把标签从 ATL_ISAR 复制过来
    src_label1_dir = os.path.join(img_ATL_ISAR_path, 'train', 'labels')
    src_label2_dir = os.path.join(img_ATL_ISAR_path, 'val', 'labels')
    dst_label1_dir = os.path.join(img_Add_Noise_path, 'train', 'labels')
    dst_label2_dir = os.path.join(img_Add_Noise_path, 'val', 'labels')
    # 把文件夹从src_label_dir复制到dst_label_dir
    shutil.copytree(src_label1_dir, dst_label1_dir)
    shutil.copytree(src_label2_dir, dst_label2_dir)
    shutil.copyfile(os.path.join(img_ATL_ISAR_path, 'train','classes.txt'), os.path.join(img_Add_Noise_path, 'train','classes.txt'))
    shutil.copyfile(os.path.join(img_ATL_ISAR_path, 'val','classes.txt'), os.path.join(img_Add_Noise_path, 'val','classes.txt'))


        
        

