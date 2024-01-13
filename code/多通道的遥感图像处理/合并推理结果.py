from PIL import Image
import os
import numpy as np
from ATL_path import scandir, mkdir_or_exist, find_data_list

from osgeo import gdal
from tqdm import trange

def main():
    big_img_path = '/share/home/aitlong/ATL/2023-GaoFen-Fusai/Dataset/test/3_image.tif'
    big_img_gdal = gdal.Open(big_img_path)
    big_img = big_img_gdal.ReadAsArray()
    big_img = big_img.transpose((1,2,0))

    h, w, c = big_img.shape
    print(f'big_img.shape:{big_img.shape}')

    new_big_img = np.zeros((h,w),dtype=np.uint8)
    print(f'new_big_img: {new_big_img.shape}')

    image_list = find_data_list('./temple_results/nanxinda/inference_3', suffix='.tif')

    for index in trange(len(image_list)):
        image_path = image_list[index]
        image = np.array(Image.open(image_path))
        # print(image_path)
        # print(f'little img shape,{image.shape}')
        if not 'xia' in image_path and not 'you' in image_path and not 'youxia' in image_path:
            img_h = int(os.path.basename(image_path).split('_')[0])
            img_w = int(os.path.basename(image_path).split('_')[1])
            # print(img_h,img_w)
            new_big_img[img_h*512:img_h*512+512,img_w*512:img_w*512+512] = image
        elif (not 'youxia' in image_path) and ('xia' in image_path):
            img_w = int(os.path.basename(image_path).split('_')[1])
            new_big_img[h-512:h, img_w*512:img_w*512+512] = image
        elif (not 'youxia' in image_path) and ('you' in image_path):
            img_h = int(os.path.basename(image_path).split('_')[1])
            new_big_img[img_h*512:img_h*512+512,w-512:w] = image
        elif 'youxia' in image_path:
            new_big_img[h-512:h,w-512:w] = image

    mkdir_or_exist('atl_inference_results')
    Image.fromarray(new_big_img).save(os.path.join('atl_inference_results','inference_3_image.tif'))

if __name__ == '__main__':
    main()
