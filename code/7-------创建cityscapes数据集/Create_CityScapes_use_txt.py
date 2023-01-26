# tools/convert_datasets/Create_CityScapes_use_txt.py

import numpy as np
from PIL import Image 
from tqdm import trange
import os

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
#       ===>>    python tools/convert_datasets/Create_CityScapes_use_txt.py  --out-dir [Your out dir]
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

data_root = 'data/cityscapes'

# txt_files = ['train.txt', 'val.txt','test.txt']
# data_files = ['train', 'val', 'test']

txt_files = ['val.txt','test.txt']
data_files = ['val', 'test']

leftImg8bit = 'leftImg8bit'
gtFine = 'gtFine'

output_path = '/HOME/scz5158/run/ATL/OpenMMLab/Dataset/cityscapes'

# create output files
if not os.path.exists(output_path): os.mkdir(output_path)
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
for txt_file, data_file in zip(txt_files, data_files):
    txt = np.loadtxt(os.path.join(data_root,txt_file),dtype='str')
    for img_index in trange(len(txt)):
        img = Image.open(os.path.join(
            data_root, leftImg8bit, data_file, txt[img_index]+'_leftImg8bit.png'))
        lable = Image.open(os.path.join(
            data_root, gtFine, data_file, txt[img_index]+'_gtFine_labelTrainIds.png'))
        
        img.save(os.path.join(output_path, leftImg8bit, data_file, 
                (txt[img_index]+'_leftImg8bit.png').split('/')[1]))
        lable.save(os.path.join(output_path, gtFine, data_file, \
                (txt[img_index]+'_gtFine_labelTrainIds.png').split('/')[1]))
