from osgeo import gdal
import numpy as np
from osgeo import osr
import os
import cv2
from ATL_path import scandir, mkdir_or_exist, find_data_list

img_path ='/share/home/aitlong/ATL/2023-GaoFen-Fusai/Dataset/test/3_image.tif'
save_path = '/share/home/aitlong/ATL/2023-GaoFen-Fusai/Dataset/test_inference3'

mkdir_or_exist(save_path)

img_gdal = gdal.Open(img_path)
img = img_gdal.ReadAsArray()
img = img.transpose((1,2,0))

h, w, c = img.shape
rows, cols, bands = img.shape

crop_size = 512


hang = h - (h//512)*512
lie =  w - (w//512)*512

print(f'可裁成{h//512}行...{hang}')
print(f'可裁成{w//512}列...{lie}')
print(f'共512*512：{((h//512)*(w//512))}张，边缘处')




# 512的部分
for i in range(h//512):
    for j in range(w//512):
        new_512 = np.zeros((512,512,4),dtype=np.uint16)
        new_512 = img[i*512:i*512+512,j*512:j*512+512,:]   #横着来
        
        out_path = os.path.join(save_path,str(i)+'_'+str(j)+'_'+'512.tif')
        # cv2.imwrite(out_path,new_512)
        Driver = gdal.GetDriverByName("Gtiff")
        new_img = Driver.Create(out_path, 512, 512, 4, gdal.GDT_UInt16)
        for band_num in range(bands):
            band = new_img.GetRasterBand(band_num+1)
            band.WriteArray(new_512[:, :, band_num])

#以外的部分

# 下边的
for j in range(w//512):
    new_512 = np.zeros((512,512,4),dtype=np.uint16)
    new_512 = img[h-512:h, j*512:j*512+512, :]   #横着来
    
    out_path = os.path.join(save_path,str('xia')+'_'+str(j)+'_'+'512.tif')
    # cv2.imwrite(out_path,new_512)
    Driver = gdal.GetDriverByName("Gtiff")
    new_img = Driver.Create(out_path, 512, 512, 4, gdal.GDT_UInt16)
    for band_num in range(bands):
        band = new_img.GetRasterBand(band_num+1)
        band.WriteArray(new_512[:, :, band_num])

#右边的
for j in range(h//512):
    new_512 = np.zeros((512,512,4),dtype=np.uint16)
    new_512 = img[j*512:j*512+512, w-512:w, :]   #横着来
    
    out_path = os.path.join(save_path,str('you')+'_'+str(j)+'_'+'512.tif')
    # cv2.imwrite(out_path,new_512)
    Driver = gdal.GetDriverByName("Gtiff")
    new_img = Driver.Create(out_path, 512, 512, 4, gdal.GDT_UInt16)
    for band_num in range(bands):
        band = new_img.GetRasterBand(band_num+1)
        band.WriteArray(new_512[:, :, band_num])

#右下角的
new_512 = np.zeros((512,512,4),dtype=np.uint16)
new_512 = img[h-512:h, w-512:w, :]   #横着来
out_path = os.path.join(save_path,str('youxia')+'_'+'512.tif')
# cv2.imwrite(out_path,new_512)
Driver = gdal.GetDriverByName("Gtiff")
new_img = Driver.Create(out_path, 512, 512, 4, gdal.GDT_UInt16)
for band_num in range(bands):
    band = new_img.GetRasterBand(band_num+1)
    band.WriteArray(new_512[:, :, band_num])

print('--- 裁切完成')
