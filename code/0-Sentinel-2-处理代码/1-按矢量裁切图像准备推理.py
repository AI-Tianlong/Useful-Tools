from ATL_Tools import mkdir_or_exist, find_data_list
from ATL_Tools.ATL_gdal import crop_tif_with_json_nan
from tqdm import tqdm
import os 

img_path_all = '../要推理的images-未裁切/'
output_path_all = '../要推理的images-矢量裁切/'
json_path_all = '../要推理的json/'
mkdir_or_exist(output_path_all)
img_list = find_data_list(img_path_all, suffix='.tif')


for img_path in tqdm(img_list, colour='Green'):

    img_output_path = os.path.join(output_path_all, os.path.basename(img_path))
    json_path = os.path.join(json_path_all, os.path.basename(img_path).split('_')[-1].replace('.tif', '.json'))
    print(f'正在裁切: {img_output_path},json: {json_path}')
    crop_tif_with_json_nan(img_path, img_output_path, json_path)
