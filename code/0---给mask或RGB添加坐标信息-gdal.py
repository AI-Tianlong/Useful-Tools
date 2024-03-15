import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from ATL_Tools import mkdir_or_exist, find_data_list
import os
import tiffile as tif
from osgeo import gdal

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

IMG_path = '/opt/AI-Tianlong/Datasets/ATL-ATLNongYe/5billion-S2/Big-image/images-origin'
RGB_path = '/opt/AI-Tianlong/Datasets/ATL-ATLNongYe/5billion-S2/inference_result_RGB'

img_lists = find_data_list(RGB_path, suffix='.png')

for img_path in tqdm(img_lists):
    img_basename = os.path.basename(img_path)

    img_gdal_ori = gdal.Open(img_path.replace('.png', '.tif'), gdal.GA_ReadOnly)
    img_gdal_label = gdal.Open(os.path.join(RGB_path, os.path.basename(img_path)))


    trans = img_gdal_ori.GetGeoTransform()
    proj = img_gdal_ori.GetProjection()

    img_gdal_label.SetGeoTransform(trans)
    img_gdal_label.SetProjection(proj)

    output_path = img_path
    gdal.Warp(output_path, img_gdal_label)

    img_gdal_ori = None
    img_gdal_label = None
