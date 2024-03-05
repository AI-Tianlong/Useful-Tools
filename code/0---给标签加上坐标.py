from ATL_Tools import mkdir_or_exist, find_data_list
from osgeo import gdal
import numpy as np 


image_paths = '/opt/AI-Tianlong/Datasets/ATL_DATASETS/Harbin/images_3channel'
label_paths = '/opt/AI-Tianlong/Datasets/ATL_DATASETS/Harbin/labels'

img_lists = find_data_list(image_paths, suffix='.tif')
label_lists = find_data_list(label_paths, suffix='.tif')

for image_path, label_path in zip(data_list, label_list):
    image = gdal.Open(image_path,  gdal.GA_ReadOnly)
    label = gdal.Open(label_path,  gdal.GA_ReadOnly)

    img_trans = image.GetGeoTransform()
    img_proj = image.GetProjection()
    
