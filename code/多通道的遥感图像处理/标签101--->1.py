#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 user <user@4029GP-TR>
#
# Distributed under terms of the MIT license.

import glob
import os

from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
# from mmengine.utils import track_parallel_progress
from mmcv import track_parallel_progress
from ATL_path import scandir, mkdir_or_exist, find_data_list

class_list = [11, 12, 21, 22, 23, 24, 31, 32, 33, 41, 43, 46, 51, 52, 53]

label_path = '/share/home/aitlong/ATL/2023-vit-adapter/ViT-Adapter/segmentation/data/tuilijieguo—final'
save_path = label_path.replace('tuilijieguo—final', 'tuilijieguo—final-label')
mkdir_or_exist(save_path)

each_label_path = glob.glob(label_path+'/*.tif')
print('--------------------------------------')
print(each_label_path)

def preprocess(each_label_path):
    label = np.array(Image.open(each_label_path))
    h , w = label.shape
    label_15 = np.ones((h, w)) * 255

    for j in range(len(class_list)):
        label_15[np.where(label == class_list[j])] = j

    label_15 = Image.fromarray(np.array(label_15, dtype=np.uint8))
    label_15.save(each_label_path.replace('tuilijieguo—final', 'tuilijieguo—final-label'))
_ = track_parallel_progress(preprocess, each_label_path, 8)
# 
# preprocess(each_label_path)
