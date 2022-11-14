import cv2
import numpy as np
import os
from tqdm import trange, tqdm
from sklearn.cross_decomposition import CCA
from PIL import  Image


def read_all_images():  #将所有标签读到内存中
    train_path = r'D:\ATL\SIRS\chusai\FuSai\Dataset\fusai_release\train\labels_18'
    test_path = r'D:\ATL\SIRS\chusai\FuSai\Dataset\fusai_release\test\labels_18'

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


if __name__ == '__main__':
    
    train_path = r'D:\ATL\SIRS\chusai\FuSai\Dataset\fusai_release\train\labels_18'
    test_path = r'D:\ATL\SIRS\chusai\FuSai\Dataset\fusai_release\test\labels_18'

    train_list= os.listdir(train_path)
    test_list = os.listdir(test_path)

    train_imgs, test_imgs = read_all_images()

    with open(r'D:\ATL\SIRS\chusai\FuSai\Dataset\fusai_release\CCA.txt', 'w+') as f:
        tbar = tqdm(test_imgs,desc=f'test:  ',colour='BLUE')
        for test_index, test_img in enumerate(tbar):

            test_hist = hist_ATL(test_img)

            CCA_list = []

            ttbar = tqdm(train_imgs,desc=f'train:  ',colour='GREEN',leave=False)
            for train_index, train_img in enumerate(ttbar):
                
                train_hist = hist_ATL(train_img)

                CCA_AandB = CCA_ATL(train_hist, test_hist)

                CCA_list.append(CCA_AandB)

            max_CCA = max(CCA_list)

            CCA_train_img_list = []
            for i in range(5):
                arg_index = np.argmax(CCA_list)
                arg_name = train_list[arg_index]

                CCA_list[arg_index] = 0
                CCA_train_img_list.append(arg_name)
            
            write_ATL = str(max_CCA)+',  '+ test_list[test_index]+',  '+CCA_train_img_list[0]+',  '+CCA_train_img_list[1]+',  '+CCA_train_img_list[2]+',  '+CCA_train_img_list[3]+',  '+CCA_train_img_list[4]+'\n'
            print(f'最大相关系数：{max_CCA}  test: {test_list[test_index]}  tran: {CCA_train_img_list[0]}  {CCA_train_img_list[1]}  {CCA_train_img_list[2]}  {CCA_train_img_list[3]}  {CCA_train_img_list[4]}')
            
            f.write(write_ATL)
            f.flush()







