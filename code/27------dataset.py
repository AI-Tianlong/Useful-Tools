import numpy as np
import torch
import torch.utils.data as data
import glob
from PIL import Image
import os
def is_image_file(filename):  # 定义一个判断是否是图片的函数
    return any(filename.endswith(extension) for extension in [".tif", '.png', '.jpg'])


class img_dataset(data.Dataset):

    def __init__(self, root_dir,img_transform=None, label_transform=None):#自定义初始化
        
        train_path = 'Train'
        test_path = 'Val'
        images_path = 'image' #文件夹
        masks_path = 'label' #文件夹
        
        super(img_dataset, self).__init__() #继承了父类的方法
        
        self.root_dir = root_dir
        self.img_transform = img_transform
        self.label_transform = label_transform
        
        self.img_path = os.path.join(self.root_dir, images_path)
        self.lab_path = os.path.join(self.root_dir, masks_path)
        
        self.images_list = os.listdir(self.img_path)#关于os包里面函数的具体说明，在第一本书的183页
                                                     #os.listdir() 方法用于返回指定的文件夹 包含的文件或文件夹的名字 的列表
                                                     #要注意的是只包含指定文件夹里面的文件名，而不是绝对路径
        self.labels_list = os.listdir(self.lab_path)
 
    
    def __getitem__(self, idx):
        img = np.array(Image.open(os.path.join(self.img_path, self.images_list[idx])))    #  读取每张图片
        if self.img_transform:
            img = self.img_transform(img)
        lab = np.array(Image.open(os.path.join(self.lab_path, self.labels_list[idx]))) 
        
        return img, lab
    
    def __len__(self):
        
        return len(self.images_list)  
    
