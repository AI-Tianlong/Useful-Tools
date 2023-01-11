# tools/convert_datasets/Create_CityScapes_use_txt.py

import numpy as np
from PIL import Image 
from tqdm import trange
import os
from multiprocessing import Pool
from multiprocessing import Process

# first step: 
#       Before run 'Create_CityScapes_use_txt.py(this python file)' 
#       You should Run this bash command first:  
#       ===>>>  python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 6  
#       will gerenate 'train.txt', 'val.txt','test.txt', 
#       Contains the names of the picture, e.g strasbourg/strasbourg_000000_025491
#       More details:
#           https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#cityscapes
#
# second step: 
#       Run this python file:
#       ===>>  python tools/convert_datasets/Create_CityScapes_use_txt_multi_progress.py  --out-dir [Your out dir]
#       will generate the final cityscapes Dataset like
#                        mmsegmentation
#                        ├── data
#                        │   ├── cityscapes
#                        │   │   ├── leftImg8bit
#                        │   │   │   ├── train
#                        │   │   │   ├── val
#                        │   │   ├── gtFine
#                        │   │   │   ├── train
#                        │   │   │   ├── val              

num_progress = 6

data_root = 'data/cityscapes_zip'

txt_files = ['train.txt', 'val.txt','test.txt']
data_files = ['train', 'val', 'test']

leftImg8bit = 'leftImg8bit'
gtFine = 'gtFine'

output_path = 'data/cityscapes'

# create output files
if not os.path.exists(output_path): 
    os.mkdir(output_path)
    for data_name in data_files:
        if not os.path.exists(os.path.join(output_path,leftImg8bit)): 
            os.mkdir(os.path.join(output_path,leftImg8bit)) 
        if not os.path.exists(os.path.join(output_path,gtFine)): 
            os.mkdir(os.path.join(output_path,gtFine)) 
        if not os.path.exists(os.path.join(output_path,leftImg8bit,data_name)): 
            os.mkdir(os.path.join(output_path,leftImg8bit,data_name)) 
        if not os.path.exists(os.path.join(output_path,gtFine,data_name)): 
            os.mkdir(os.path.join(output_path,gtFine,data_name)) 

# create cityscapes Dataset
def create_cityscapes_multi(jobi):
    for txt_file, data_file in zip(txt_files, data_files):
        num=1
        txt = np.loadtxt(os.path.join(data_root,txt_file),dtype='str')
        for img_index in range(jobi*(len(txt)//num_progress), (jobi+1)*(len(txt)//num_progress)):  #
            img = Image.open(os.path.join(
                data_root, leftImg8bit, data_file, txt[img_index]+'_leftImg8bit.png'))
            lable = Image.open(os.path.join(
                data_root, gtFine, data_file, txt[img_index]+'_gtFine_labelTrainIds.png'))
            
            img.save(os.path.join(output_path, leftImg8bit, data_file, 
                    (txt[img_index]+'_leftImg8bit.png').split('/')[1]))
            lable.save(os.path.join(output_path, gtFine, data_file, \
                    (txt[img_index]+'_gtFine_labelTrainIds.png').split('/')[1]))
            num += 1
            print(f'已完成{num*6}张')

def create_cityscapes_single(num_progress):
    for txt_file, data_file in zip(txt_files, data_files):
        num=1
        txt = np.loadtxt(os.path.join(data_root,txt_file),dtype='str')
        for img_index in range((len(txt)//num_progress)*num_progress,len(txt)):  #
            img = Image.open(os.path.join(
                data_root, leftImg8bit, data_file, txt[img_index]+'_leftImg8bit.png'))
            lable = Image.open(os.path.join(
                data_root, gtFine, data_file, txt[img_index]+'_gtFine_labelTrainIds.png'))
            
            img.save(os.path.join(output_path, leftImg8bit, data_file, 
                    (txt[img_index]+'_leftImg8bit.png').split('/')[1]))
            lable.save(os.path.join(output_path, gtFine, data_file, \
                    (txt[img_index]+'_gtFine_labelTrainIds.png').split('/')[1]))
            num += 1
            print(f'已完成{num*6}张')

        
def main():
    ps=[]

    for i in range(num_progress):
        p=Process(target=create_cityscapes_multi,name="create_cityscapes"+str(i),args=(i,))
        ps.append(p)
    # 开启进程
    for i in range(num_progress):
        ps[i].start()
    # 阻塞进程
    for i in range(num_progress):
        ps[i].join()
    print("多进程任务结束，开始最后一个进程")
    create_cityscapes_single(num_progress)

if __name__ == '__main__':
    main()
