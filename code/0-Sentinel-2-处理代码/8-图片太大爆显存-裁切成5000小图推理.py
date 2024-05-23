# Copyright (c) OpenMMLab. All rights reserved.
from osgeo import gdal
import argparse
import os
import os.path as osp
from PIL import Image
import numpy as np

from tqdm import tqdm
from ATL_Tools import mkdir_or_exist, find_data_list

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert potsdam dataset to mmsegmentation format')
    parser.add_argument('-d', '--dataset_path', help='potsdam folder path', 
                        default='../要推理的images-矢量裁切/')
    parser.add_argument('-t', '--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path',
                        default='../要推理的images-矢量裁切-小图/')
    parser.add_argument(
        '--crop_size',
        type=int,
        help='clipped size of image after preparation',
        default=5000)
    args = parser.parse_args()
    return args


def clip_big_image(image_path, save_path, crop_size=8000):
    # Original image of Potsdam dataset is very large, thus pre-processing
    # of them is adopted. Given fixed clip size and stride size to generate
    # clipped image, the intersection　of width and height is determined.
    # For example, given one 5120 x 5120 original image, the clip size is
    # 512 and stride size is 256, thus it would generate 20x20 = 400 images
    # whose size are all 512x512.

    #====================
    img_gdal = gdal.Open(image_path)
    img_bit = img_gdal.GetRasterBand(1).DataType

    img_basename = osp.basename(image_path).split('.')[0] #nangangqu.tif
    
    # 对图像进行裁切,分为8000*8000的地方和512*512的地方
    image_gdal = gdal.Open(image_path)
    img = image_gdal.ReadAsArray()
    img = img.transpose((1,2,0))

    h, w, c = img.shape
    rows, cols, bands = img.shape

    if h < crop_size or w < crop_size:
        print(f'--- 当前 {img_basename} 图像尺寸小于 {crop_size}，不进行裁切')

        out_path = os.path.join(save_path, osp.basename(image_path))
        Driver = gdal.GetDriverByName("Gtiff")
        new_img = Driver.Create(out_path, w,h,c, img_bit)
        for band_num in range(bands):
            band = new_img.GetRasterBand(band_num+1)
            band.WriteArray(img[:, :, band_num])
        return None
    
    else:
    
        hang = h - (h//crop_size)*crop_size
        lie =  w - (w//crop_size)*crop_size
        print(f'图像尺寸：{img.shape}')
        print(f'可裁成{h//crop_size+1}行...{hang}')
        print(f'可裁成{w//crop_size+1}列...{lie}')
        print(f'共{crop_size}：{((h//crop_size+1)*(w//crop_size+1))+1}张')

        # 8000的部分 xxxxx._0_0_8000.tif
        for i in range(h//crop_size):
            for j in range(w//crop_size):
                out_path = os.path.join(save_path, img_basename+'_'+str(i)+'_'+str(j)+'_'+str(crop_size)+'.tif')
                Driver = gdal.GetDriverByName("Gtiff")

                new_512 = np.zeros((crop_size,crop_size, c),dtype=np.uint8)
                new_img = Driver.Create(out_path, crop_size, crop_size, c, img_bit)
            
                new_512 = img[i*crop_size:i*crop_size+crop_size,j*crop_size:j*crop_size+crop_size,:]   #横着来       

                for band_num in range(bands):
                    band = new_img.GetRasterBand(band_num + 1)
                    band.WriteArray(new_512[:, :, band_num])

        #以外的部分

        # 下边的 xxxxx._xia_0_8000.tif
        for j in range(w//crop_size):
            new_512 = np.zeros((crop_size,crop_size, c),dtype=np.uint8)
            new_512 = img[h-crop_size:h, j*crop_size:j*crop_size+crop_size, :]   #横着来
            
            out_path = os.path.join(save_path, img_basename+'_'+str('xia')+'_'+str(j)+'_'+str(crop_size)+'.tif')
            # cv2.imwrite(out_path,new_512)
            Driver = gdal.GetDriverByName("Gtiff")
            new_img = Driver.Create(out_path, crop_size,crop_size, c, img_bit)
            for band_num in range(bands):
                band = new_img.GetRasterBand(band_num+1)
                band.WriteArray(new_512[:, :, band_num])

        #右边的 xxxxx._you_0_8000.tif
        for j in range(h//crop_size):
            new_512 = np.zeros((crop_size,crop_size, c),dtype=np.uint8)
            new_512 = img[j*crop_size:j*crop_size+crop_size, w-crop_size:w, :]   #横着来
            
            out_path = os.path.join(save_path,img_basename+'_'+str('you')+'_'+str(j)+'_'+str(crop_size)+'.tif')
            # cv2.imwrite(out_path,new_512)
            Driver = gdal.GetDriverByName("Gtiff")
            new_img = Driver.Create(out_path, crop_size,crop_size, c, img_bit)
            for band_num in range(bands):
                band = new_img.GetRasterBand(band_num+1)
                band.WriteArray(new_512[:, :, band_num])

        #右下角的
        new_512 = np.zeros((crop_size,crop_size, c),dtype=np.uint8)
        new_512 = img[h-crop_size:h, w-crop_size:w, :]   #横着来
        out_path = os.path.join(save_path, img_basename+'_'+str('youxia')+'_'+str(crop_size)+'.tif')
        # cv2.imwrite(out_path,new_512)
        Driver = gdal.GetDriverByName("Gtiff")
        new_img = Driver.Create(out_path, crop_size,crop_size, c, img_bit)
        for band_num in range(bands):
            band = new_img.GetRasterBand(band_num+1)
            band.WriteArray(new_512[:, :, band_num])

    print(f'--- 当前 {img_basename} 图像裁切完成')

def main():
    args = parse_args()
    dataset_path = args.dataset_path
    data_list = find_data_list(dataset_path, suffix='.tif')
    assert len(data_list) > 0, f'Found no images in {dataset_path}.'

    if args.out_dir is None:
        out_dir = os.path.join(args.dataset_path, 'data', 'Harbin_8000')
    else:   
        out_dir = args.out_dir
    
    mkdir_or_exist(out_dir)
    # mkdir_or_exist(osp.join(out_dir, 'images'))
    # mkdir_or_exist(osp.join(out_dir, 'labels'))

    print(f'共找到数据：{data_list} 张')

    # prog_bar = ProgressBar(len(src_path_list))
    for img_path in tqdm(data_list, desc=f"---正在切分图像",colour="GREEN"):
        clip_big_image(img_path, out_dir, crop_size=args.crop_size)
        # prog_bar.update()

    final_img_data_list = find_data_list(out_dir, suffix='.tif')
    print(f'共生成数据：{len(final_img_data_list)} 张')
    print('Done!')


if __name__ == '__main__':
    main()
