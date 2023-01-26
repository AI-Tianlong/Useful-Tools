import cv2
import numpy as np
import os
from tqdm import trange, tqdm
from sklearn.cross_decomposition import CCA
from PIL import  Image
import threading
import time
import os,time
from multiprocessing import Pool
from multiprocessing import Process


def read_all_images():  #将所有标签读到内存中
    train_path = r'/share/home/dongzhe/ATL/Baseline/fusai_release/train/labels_18'
    test_path = r'/share/home/dongzhe/ATL/Baseline/fusai_release/test/78.84_18'

    train_list= os.listdir(train_path)
    test_list = os.listdir(test_path)

    train_imgs, test_imgs = [], []

    #读取训练集到内存
    for index in trange(len(train_list),desc='读取训练集label至内存ing：  ',colour='GREEN'):
        img = np.asarray(Image.open(os.path.join(train_path,train_list[index])), dtype=np.float32)
        train_imgs.append(img)
    
    #读取测试集到内存
    for index in trange(len(test_list),desc='读取测试集集label至内存ing：  ',colour='BLUE'):
        img = np.asarray(Image.open(os.path.join(test_path,test_list[index])), dtype=np.float32)
        test_imgs.append(img)
    
    return train_imgs, test_imgs

def read_some_images():  #读取一些固定的index的到内存中
    train_path = r'/share/home/dongzhe/ATL/Baseline/fusai_release/train/labels_18'
    test_path = r'/share/home/dongzhe/ATL/Baseline/fusai_release/test/78.84_18'

    train_list= os.listdir(train_path)
    test_list = os.listdir(test_path)

    train_imgs, test_imgs = [], []

    #读取训练集到内存
    for index in trange(len(train_list),desc='读取训练集label至内存ing：  '):
        img = np.asarray(Image.open(os.path.join(train_path,train_list[index])), dtype=np.float32)
        train_imgs.append(img)
    
    #读取测试集到内存
    for index in trange(len(test_list),desc='读取测试集集label至内存ing：  '):
        img = np.asarray(Image.open(os.path.join(test_path,test_list[index])), dtype=np.float32)
        test_imgs.append(img)
    
    return train_imgs, test_imgs


def CCA_ATL(hist1,hist2):   #计算两个标签的相关系数

    # 建立模型
    cca = CCA(n_components=1)
    #如果想计算第二主成分对应的相关系数cca = CCA(n_components=2)
    # 训练数据
    cca.fit(hist1, hist2)
    # 降维操作
    # print(X)
    X_train_r, Y_train_r = cca.transform(hist1, hist2)
    # print(X_train_r)
    # print(np.corrcoef(X_train_r[:, 0], Y_train_r[:, 0])[0, 1]) #输出相关系数
    #如果想计算第二主成分对应的相关系数 print(np.corrcoef(X_train_r[:, 1], Y_train_r[:, 1])[0, 1])
    CCA_ = np.corrcoef(X_train_r[:, 0], Y_train_r[:, 0])[0, 1]

    return CCA_


def hist_ATL(img):   #统计hist

    hist = cv2.calcHist([img], [0], None, [18], [0,17])

    return hist


import threading
import time

def shaixuan(jobi):

    print("子进程执行中>>> pid={0},ppid={1}".format(os.getpid(),os.getppid()))
    train_imgs, test_imgs = read_all_images()
    train_path = r'/share/home/dongzhe/ATL/Baseline/fusai_release/train/labels_18'
    test_path = r'/share/home/dongzhe/ATL/Baseline/fusai_release/test/78.84_18'

    train_list= os.listdir(train_path)
    test_list = os.listdir(test_path)

    txt_name = '/share/home/dongzhe/ATL/筛选数据/CCA_'+str(jobi)+'.txt'

    index_fanwei = np.array([0,292]) + jobi*293


    with open(txt_name, 'w+') as f:
        
        for test_index in range(len(test_list)):

            if index_fanwei[0]<=test_index and test_index<=index_fanwei[1]:
                test_hist = hist_ATL(test_imgs[test_index])

                CCA_list = []

                ttbar = tqdm(train_imgs,desc=f'train:  ',colour='GREEN',leave=False)
                for train_index, train_img in enumerate(ttbar):
                    
                    train_hist = hist_ATL(train_img)

                    CCA_AandB = CCA_ATL(train_hist, test_hist)

                    CCA_list.append(CCA_AandB)

                max_CCA = max(CCA_list)

                CCA_train_img_list = []
                
                f.write('测试集：'+test_list[test_index]+'训练集：')
                for i in range(50):
                    arg_index = np.argmax(CCA_list)
                    arg_name = train_list[arg_index]

                    CCA_list[arg_index] = 0
                    CCA_train_img_list.append(arg_name)
                    
                    f.write(',  '+arg_name)
                f.write('\n')
                f.flush()
            else:
                pass
    print("子进程终止>>> pid={0}".format(os.getpid()))

def main():
    print("主进程执行中>>> pid={0}".format(os.getpid()))
    
    ps=[]

    for i in range(14):
        p=Process(target=shaixuan,name="shaixuan"+str(i),args=(i,))
        ps.append(p)
    # 开启进程
    for i in range(14):
        ps[i].start()
    # 阻塞进程
    for i in range(14):
        ps[i].join()
    print("主进程终止")


if __name__ == '__main__':
    main()

# if __name__=='__main__':
#   #多线程
#     train_imgs, test_imgs = read_all_images()

#     for jobi in range(14):#利用循环创建5个线程  4102/293=14   第一个0-292
#         fanwei = np.array([0,292])+jobi*293 
#         t=threading.Thread(target=shaixuan,args=(fanwei,jobi,))
#         print(len(threading.enumerate()))  #查看线程数量和进程数量总和
#         #启动线程
#         t.start()














# if __name__ == '__main__':
    
#     train_path = r'/share/home/dongzhe/ATL/Baseline/fusai_release/train/labels_18'
#     test_path = r'/share/home/dongzhe/ATL/Baseline/fusai_release/test/78.84_18'

#     train_list= os.listdir(train_path)
#     test_list = os.listdir(test_path)

#     train_imgs, test_imgs = read_all_images()

#     with open(r'/share/home/dongzhe/ATL/筛选数据/CCA.txt', 'w+') as f:
#         tbar = tqdm(test_imgs,desc=f'test:  ',colour='BLUE')
#         for test_index, test_img in enumerate(tbar):

#             test_hist = hist_ATL(test_img)

#             CCA_list = []

#             ttbar = tqdm(train_imgs,desc=f'train:  ',colour='GREEN',leave=False)
#             for train_index, train_img in enumerate(ttbar):
                
#                 train_hist = hist_ATL(train_img)

#                 CCA_AandB = CCA_ATL(train_hist, test_hist)

#                 CCA_list.append(CCA_AandB)

#             max_CCA = max(CCA_list)

#             CCA_train_img_list = []
            
#             f.write(test_list[test_index])
#             for i in range(50):
#                 arg_index = np.argmax(CCA_list)
#                 arg_name = train_list[arg_index]

#                 CCA_list[arg_index] = 0
#                 CCA_train_img_list.append(arg_name)
                
#                 f.write(',  '+arg_name)
#             f.write('\n')
#             f.flush()
            
#             # print(f'最大相关系数：{max_CCA}  test: {test_list[test_index]}  tran: {CCA_train_img_list[0]}  {CCA_train_img_list[1]}  {CCA_train_img_list[2]}  {CCA_train_img_list[3]}  {CCA_train_img_list[4]}')
    
