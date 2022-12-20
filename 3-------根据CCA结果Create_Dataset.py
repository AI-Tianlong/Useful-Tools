import numpy as np
import os
from tqdm import trange, tqdm
from PIL import  Image
import pandas as pd


CCA_Path = r'D:\ATL\SIRS\chusai\FuSai\Dataset\fusai_release\CCA.txt'

data = pd.read_table(CCA_Path, sep = ',  ', header = None, names = ['CCA','test','train-1','train-2','train-3','train-4','train-5'])

train_imgs_pd = data[['train-1','train-2','train-3','train-4','train-5']]

img_list = np.array(train_imgs_pd)

res = set(img_list.flatten().tolist())
print(len(res))

img_path = r'D:\ATL\SIRS\chusai\FuSai\Dataset\fusai_release\train\labels_18'
img_save_path = r'D:\ATL\SIRS\chusai\FuSai\Dataset\fusai_release\CCA_train\labels_18'

index = 0 
for i in trange(img_list.shape[0]):
    for j in range(img_list.shape[1]):
        img = np.array(Image.open(os.path.join(img_path,img_list[i,j].replace('.png','.png'))))
        IMG = Image.fromarray(img)
        IMG.save(os.path.join(img_save_path,str(index)+'_'+img_list[i,j].replace('.png','.png')))
        index += 1

