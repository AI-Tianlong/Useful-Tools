import os 
import numpy as np
from path import scandir, find_data_list, mkdir_or_exist
from tqdm import trange, tqdm
import cv2
from typing import List
from PIL import Image
from pandas import read_csv
import json


# =====================================需要修改的一些参数===================================================

img_path = '/media/atl/Data1/ATL/AI_work/OpenMMLab/mmyolo/data/ATL_ISAR/val/vis_test'   # 原始数据集的根目录
label_path = '/media/atl/Data1/ATL/AI_work/OpenMMLab/mmyolo/work_dirs/ATL_yolov8_s_ATL_ISAR/bbox.json'

colormap = {'ZhanDouJi':(0, 255, 0),'YunShuJi':(132, 112, 255), 'BaJi':(0, 191, 255)}  # 色盘，可根据类别添加新颜色
class_names ={'0':'ZhanDouJi', '1':'YunShuJi', '2':'BaJi'}

def draw_label_type():

    # out_path = os.path.join(img_root_path, dataset_type, 'vis_test')
    # mkdir_or_exist(out_path)

    # img_list = os.listdir(os.path.join(img_root_path,dataset_type,'images'))
    # label_list = os.listdir(os.path.join(img_root_path,dataset_type,'labels'))
    with open(label_path,'r') as f:
        labels = json.load(f)

        for i in trange(2, desc=f'正在创建可视化 val ', colour='GREEN'):
            image_whole_path = os.path.join(img_path, str(labels[i]['image_id'])+'.png')

            # 读取图像文件
            draw_img = cv2.imread(image_whole_path)
            # 读取 labels

            
            labels_bbox = labels[i]['bbox']
            labels_bbox = np.array(labels_bbox,np.int32)
            labels_name = class_names[str(labels[i]['category_id'])]
            # print('---------------------------------------------')
            # print(labels_bbox)
            # print(labels_name)

            cv2.rectangle(draw_img, (labels_bbox[0], labels_bbox[1]), (labels_bbox[0]+labels_bbox[2], labels_bbox[1]+labels_bbox[3]), color=colormap[labels_name]) # 画标签框
            
            labelSize = cv2.getTextSize(labels_name + '0', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]  # 文字框占多大
            cv2.rectangle(draw_img, (labels_bbox[0], labels_bbox[1]-labelSize[1]), (labels_bbox[0]+labelSize[0], labels_bbox[1]),\
                        color=colormap[labels_name],thickness=-1) # 画标签框
            cv2.putText(draw_img, labels_name, (labels_bbox[0], labels_bbox[1]-labelSize[1]+10),\
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0)) # 写类别名
            
            Image.fromarray(draw_img).show()
            
    
        # cv2.imwrite(image_whole_path, draw_img)
    # return draw_img
    # cv2.imshow('img',draw_img)
    # cv2.waitKey(0)

if __name__ == '__main__':
    # draw_label_type(img_root_path=img_root_path, dataset_type='train')
    draw_label_type()


        
        

