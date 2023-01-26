'''
该程序可以直接将三位数的标签，转换为RGB进行可视化

'''
#============================ #数据集的地址 只需要修改这个地方 =========================
masks_file_path = r'D:\ATL\SIRS\chusai\FuSai\Dataset\fusai_release\train\labels_18'     #三位数标签路径
vis_file_path   = r'D:\ATL\SIRS\chusai\FuSai\Dataset\fusai_release\train\labels_18_RGB'  #转为RGB后的标签存储路径,程序会自动创建文件夹！
#=======================================================================================

import numpy as np
from PIL import Image
import glob
import time
from tqdm import tqdm,trange
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if not os.path.exists(vis_file_path):
    os.mkdir(vis_file_path)

#=============================================  Mask To RGB====================================
#=============  类别数 RGB ==================
COLOR_MAP = dict(
    IGNORE=(0, 0, 0),             #0-背景-黑色   ，在测评的时候会被忽视掉
    Water=(0, 0, 255),            #101--水体
    Road=(128, 128, 128),         #202--道路
    Building=(255,128,128),       #303--建筑
    Airport=(192,192,192),        #204--机场
    TrainStation=(145,204,117),   #205--火车场
    GuangFu=(80,100,115),         #806--光伏
    Parking=(70,70,70),           #807--停车场
    Gym=(255,74,74),              #808--操场
    Agricultural=(255, 255, 0),   #409--普通耕地
    DaPeng=(234,234,234),         #410--农业大棚
    GrassLand=(0, 255, 0),        #511--自然草地
    Human_GrassLand=(183,255,183),#512--绿地绿化
    Forest=(0,128,0),             #613--自然林
    Human_Forest=(0,190,0),       #614--人工林
    Barren=(128,64,0),            #715--自然裸土
    Human_Barren=(255,128,0),     #716--人为裸土
    Others=(255, 0, 0)            #817--无法确定的
    )

def render(mask_17_img, vis_path):  #(3位数标签变为2位数之后的img，np格式)
    new_mask = mask_17_img.astype(np.uint8) #把原来的图片读出来，存成uint8的格式
    cm = np.array(list(COLOR_MAP.values())).astype(np.uint8) #取出COLOR_MAP中的值，存到列表
    color_img = cm[new_mask]  #这里就变成1024*1024*3的了？ why？？
    color_img = Image.fromarray(np.uint8(color_img)) #从np的array格式转换成PIL格式
    color_img.save(vis_path)
    
#数据集的名字列表
masks_list = os.listdir(masks_file_path) #把这个文件夹中的图片的地址，以有序的列表形式存放 
print(len(masks_list))

class_list = [000,101, 202, 303, 204, 205, 806, 807, 808, 409, 410, 511, 512, 613, 614, 715, 716, 817]

if __name__ == '__main__':
    print('vis_png 转换ing...')
    
    for index in trange(len(masks_list)):
        label = np.array(Image.open(os.path.join(masks_file_path, masks_list[index])))
        # for j in range(len(class_list)):
        #     label[np.where(label==class_list[j])] = j

        render(label,os.path.join(vis_file_path,masks_list[index])) #训练集是43 测试集是45