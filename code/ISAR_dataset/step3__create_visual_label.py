import os 
import numpy as np
from path import scandir, find_data_list, mkdir_or_exist
from tqdm import trange, tqdm
import cv2
from typing import List
from PIL import Image
from pandas import read_csv


# =====================================需要修改的一些参数===================================================

img_root_path = '../Dataset_test/ATL_ISAR'   # 原始数据集的根目录


colormap = {'ZhanDouJi':(0, 255, 0),'YunShuJi':(132, 112, 255), 'BaJi':(0, 191, 255)}  # 色盘，可根据类别添加新颜色

def draw_label_type(img_root_path, dataset_type = 'train', colormap=colormap):

    out_path = os.path.join(img_root_path, dataset_type, 'vis')
    mkdir_or_exist(out_path)

    img_list = os.listdir(os.path.join(img_root_path,dataset_type,'images'))
    label_list = os.listdir(os.path.join(img_root_path,dataset_type,'labels'))

    for i in trange(len(img_list), desc=f'正在创建可视化 {dataset_type} ', colour='GREEN'):
        image_whole_path = os.path.join(img_root_path,dataset_type,'images', img_list[i])
        label_whole_path = os.path.join(img_root_path,dataset_type,'labels', label_list[i])

        # 读取图像文件
        draw_img = cv2.imread(image_whole_path)
        # 读取 labels
        labels_file = read_csv(label_whole_path)
        labels = labels_file.values

        labels_img_path = labels[:, 0]
        labels_bbox = labels[:, 1:6]
        labels_name = labels[:, 5]
        
        for j in range(labels_bbox.shape[0]):
            bbox = labels_bbox[j, :]

            class_name = bbox[-1]  # 标签的名字
            cv2.rectangle(draw_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=colormap[class_name]) # 画标签框
            
            labelSize = cv2.getTextSize(class_name + '0', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]  # 文字框占多大
            cv2.rectangle(draw_img, (bbox[0], bbox[1]-labelSize[1]), (bbox[0]+labelSize[0], bbox[1]),\
                        color=colormap[class_name],thickness=-1) # 画标签框
            cv2.putText(draw_img, class_name, (bbox[0], bbox[1]-labelSize[1]+10),\
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0)) # 写类别名
        
        cv2.imwrite(os.path.join(out_path,  img_list[i]), draw_img)
    # return draw_img
    # cv2.imshow('img',draw_img)
    # cv2.waitKey(0)

if __name__ == '__main__':
    draw_label_type(img_root_path=img_root_path, dataset_type='train')
    draw_label_type(img_root_path=img_root_path, dataset_type='val')


        
        

