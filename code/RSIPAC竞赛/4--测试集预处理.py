import os
from tqdm import trange
from PIL import Image
from PIL import ImageEnhance
import numpy as np
from tqdm import trange
from matplotlib import pyplot as plt
import cv2


test_path = '/test/images'
test_save_path = '/workspace/segmentation/Dataset/test/images'
if not os.path.exists(test_save_path):os.mkdir(test_save_path)

test_list = os.listdir(test_path)

def yuchuli1():
    for i in trange(len(test_list)):
        img = Image.open(os.path.join(test_path, test_list[i]))
        enh_con = ImageEnhance.Contrast(img)
        contrast = 1.5
        img_contrasted = enh_con.enhance(contrast)

        img_contrasted.save(os.path.join(test_save_path, test_list[i]))

def Canny(img_gray,lowThreshold):
    ratio = 3                
    kernel_size = 3  
    
    detected_edges = cv2.GaussianBlur(img_gray,(3,3),0) #高斯滤波 
    detected_edges = cv2.Canny(detected_edges,
            lowThreshold,
            lowThreshold*ratio,
            apertureSize = kernel_size)  #边缘检测

    return detected_edges

def yuchuli2():
    for img_index in trange(len(wrong_test_list)):
    
        count_thrhold=600    
        img_size = 512
        flag='down'   
        flag_down=0   
        canny_max=70
        canny_min=25

        img = cv2.imread(os.path.join(img_path,wrong_test_list[img_index]))       

        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                  
        img_after_canny_90 =  Canny(img_gray,canny_max)
        img_after_canny_25 =  Canny(img_gray,canny_min)
    

        line_xy_list=[]
        for w in range(img_size):
            temp_xy=[]
            for h in range(img_size):   
                if img_after_canny_90[h,w] != 0:   
            
                    line_xy_list.append([h,w])   
                    break    
        
        line_index=0
        for w in range(img_size):
            for h in range(img_size):   
                
                if img_after_canny_25[h,w] > 0 and (h>=line_xy_list[line_index][0] or w>=line_xy_list[line_index][1]): #>当前这个像素的值大于0，位置为线的下面的话：
                    flag_down += 1     
                elif img_after_canny_25[h,w] > 0 and h==line_xy_list[line_index][0] and w==line_xy_list[line_index][1]:  #当前这个像素的值大于0，位置为线的第一个像素的位置的话：
                    line_index += 1

                if flag_down>=count_thrhold:   
                    flag='up'
                    break
            if flag=='up':
                break
                    

        img = np.array(Image.open(os.path.join(img_path,wrong_test_list[img_index])))

        if flag=='up':   
            line_index=0

            if(line_xy_list[line_index][1]>=0 and line_xy_list[len(line_xy_list)-1][1]==511): 
                for w in range(img_size): 
                    for h in range(img_size):   
                        
                        if h==line_xy_list[line_index][0] and w==line_xy_list[line_index][1]:
                            line_index+=1
                            break
                    
                        elif (h <= line_xy_list[line_index][0] and w <=line_xy_list[line_index][1]):   
                            img[h,w,0]=0   
                            img[h,w,1]=0
                            img[h,w,2]=0
            
            elif(line_xy_list[line_index][1]==0 and line_xy_list[len(line_xy_list)-1][1]<511):
                for w in range(img_size): 
                    for h in range(img_size):   
                        
                        if line_index==len(line_xy_list):
                            img[h,w,0]=0   
                            img[h,w,1]=0
                            img[h,w,2]=0
                        
                        elif h==line_xy_list[line_index][0] and w==line_xy_list[line_index][1]:
                            line_index+=1
                            break
                        
                        elif (h < line_xy_list[line_index][0] and w <=511):   
                            img[h,w,0]=0   
                            img[h,w,1]=0
                            img[h,w,2]=0                                               

        elif flag=='down': 
            line_index=0
            for w in range(img_size): 
                for h in range(img_size):   
                    h=img_size-h-1

                    if h > line_xy_list[line_index][0] and w<=line_xy_list[line_index][1]:   
                        img[h,w,0]=0.   
                        img[h,w,1]=0.
                        img[h,w,2]=0.

                    elif h==line_xy_list[line_index][0] and w==line_xy_list[line_index][1]:
                        line_index+=1
                        break
        img = Image.fromarray(img)
        img.save(os.path.join(img_save_path,wrong_test_list[img_index]))


wrong_test_list = ['1197.tif', '124.tif', '2672.tif', '2816.tif','2840.tif', '3238.tif', '3293.tif', '3769.tif',
                    '4006.tif', '6.tif', '610.tif', '652.tif','664.tif', '799.tif', '805.tif']

img_path = '/test/images'
img_save_path = '/workspace/segmentation/Dataset/test/images'

if __name__ == '__main__':

    print('\n')
    print(f'-----------第 4 步  测试集预处理---------------')
    print('\n')
    yuchuli1()
    yuchuli2()

    print(f'---第 4 步 已完成')
