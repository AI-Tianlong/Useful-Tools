from ATL_Tools import mkdir_or_exist, find_data_list
from ATL_Tools.ATL_gdal import Mosaic_all_imgs
from tqdm import tqdm
import os 

img_path_all = '/opt/AI-Tianlong/Datasets/ATL-ATLNongYe/ATL推理大图/黑河/推理结果-3类作物/推理结果-RGB-crop'
output_path = '/opt/AI-Tianlong/Datasets/ATL-ATLNongYe/ATL推理大图/黑河/推理结果-3类作物/黑河市-3类作物-底图.tif'

Mosaic_all_imgs(img_file_path = img_path_all, 
                output_path = output_path, 
                nan_or_zero='nan', 
                output_band_chan=3,
                add_alpha_chan = True) # 32位图像不支持alpha通道,改为False
