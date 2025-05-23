import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os

from ATL_Tools import find_data_list, mkdir_or_exist, setup_logger
from ATL_Tools.ATL_gdal import save_array_to_tif 
from tqdm import tqdm 
from osgeo import gdal

atl_logger = setup_logger(show_file_path=False)
METAINFO = dict(
    classes=('Paddy field', 'Other Field', 'Forest', 'Natural meadow',
                'Artificial meadow', 'River', 'Lake', 'Pond',
                'Factory-Storage-Shopping malls', 'Urban residential',
                'Rural residential', 'Stadium', 'Park Square', 'Road',
                'Overpass', 'Railway station', 'Airport', 'Bare land',
                'Glaciers Snow'),
    palette=[[0,   240, 150], [150, 250, 0  ], [0,   150, 0  ], [250, 200, 0  ],
                [200, 200, 0  ], [0,   0,   200], [0,   150, 200], [150, 200, 250],
                [200, 0,   0  ], [250, 0,   150], [200, 150, 150], [250, 200, 150],
                [150, 150, 0  ], [250, 150, 150], [250, 150, 0  ], [250, 200, 250],
                [200, 150, 0  ], [200, 100, 50 ], [255, 255, 255]])                              
                    

def Mask2rgb_Without_Geo(mask_label_path: str, 
                         METAINFO: dict, 
                         reduce_zero_label: bool=True, 
                         save_rgb_label:bool=False,
                         save_rgb_path:str=None,
                         save_rgb_suffix:str=None,
                         add_geo:bool=False,
                         geo_img_path:str=None,
                         geo_img_suffix:str='.tif'):
    """将mask标签，根据METAINFO转为RGB可视化图像
    
    Args:
        label_path (str): mask标签路径
        METAINFO (dict): 包含类别和调色板信息
        reduce_zero_label (bool, optional): 是否包含0类，classes和palette里没包含. Defaults to True.
        add_geo (bool, optional): 是否添加地理信息. Defaults to False.

    Returns:
        np.ndarray: RGB可视化图像
    """
    palette = METAINFO['palette']
    if reduce_zero_label:
        new_palette = [[0, 0, 0]] + palette
        # print(f"palette: {new_palette}")
    else:
        new_palette = palette
        # print(f"palette: {new_palette}")

    new_palette = np.array(new_palette)
    
    
    if add_geo:
        backend = 'gdal'
    else:
        backend = 'PIL'

    # import pdb; pdb.set_trace()
    if backend == 'PIL':
        mask_label = np.array(Image.open(mask_label_path)).astype(np.uint8)
        mask_label[mask_label == 255] = 0  #真实标签中有255, 转为0
        h,w = mask_label.shape
        RGB_label = new_palette[mask_label].astype(np.uint8)
        if save_rgb_label:
            assert save_rgb_path is not None, "当需要单独保存 RGB 可视化标签时, 请提供保存路径"
            file_name, suffix = os.path.splitext(mask_label_path)
            if save_rgb_suffix is None:
                save_rgb_suffix = suffix
            output_path = os.path.join(save_rgb_path, os.path.basename(file_name) + save_rgb_suffix)
            RGB_label = Image.fromarray(RGB_label).save(output_path)
        return RGB_label
    
    elif backend == 'gdal':
        mask_label = gdal.Open(mask_label_path).ReadAsArray()
        mask_label[mask_label == 255] = 0  #真实标签中有255, 转为0
        h,w = mask_label.shape # (594, 594)
        RGB_label = new_palette[mask_label].astype(np.uint8)

        if save_rgb_label:

            assert save_rgb_path is not None, "当需要单独保存 RGB 可视化标签时, 请提供保存路径"
            file_name, suffix = os.path.splitext(mask_label_path)
            file_name = file_name.replace('_18label', '')
            geo_img_path_ = os.path.join(geo_img_path, os.path.basename(file_name)+geo_img_suffix)
            geo_img_ds = gdal.Open(geo_img_path_, gdal.GA_ReadOnly)
            geo_img_np = geo_img_ds.ReadAsArray().transpose(1, 2, 0)  #(594,594,4)  # 这里是366附近吧
            # import pdb; pdb.set_trace()
            nodata_pos = np.all(geo_img_np == [0]*geo_img_np.shape[2], axis=-1)  # 这是nodata的数值
            if RGB_label.shape != geo_img_np.shape:
                RGB_label = cv2.resize(RGB_label, (geo_img_np.shape[1], geo_img_np.shape[0]), interpolation=cv2.INTER_NEAREST)

            RGB_label[nodata_pos] = [0, 0, 0]

            if save_rgb_suffix is None:
                save_rgb_suffix = '.tif'
                # import pdb; pdb.set_trace()
            output_path = os.path.join(save_rgb_path, os.path.basename(file_name)+save_rgb_suffix)
            
            driver = gdal.GetDriverByName('GTiff')
            RGB_label_gdal = driver.Create(output_path, w, h, 3, gdal.GDT_Byte)
            RGB_label_gdal.GetRasterBand(1).WriteArray(RGB_label[:,:,0])
            RGB_label_gdal.GetRasterBand(2).WriteArray(RGB_label[:,:,1])
            RGB_label_gdal.GetRasterBand(3).WriteArray(RGB_label[:,:,2])

            if add_geo:
                assert geo_img_path is not None, "当需要添加地理信息时, 请提供地理信息图像路径 `geo_img_path`!!!"
                geo_img_path_ = os.path.join(geo_img_path, os.path.basename(file_name)+geo_img_suffix)
                geo_img_ds = gdal.Open(geo_img_path_, gdal.GA_ReadOnly)
                assert  geo_img_ds is not None, f"无法打开 {geo_img_path_}, 请检查路径及后缀"

                trans = geo_img_ds.GetGeoTransform()
                proj = geo_img_ds.GetProjection()

                RGB_label_gdal.SetGeoTransform(trans)
                RGB_label_gdal.SetProjection(proj)

            RGB_label_gdal = None
        return RGB_label

def Overlay_picture(mask_label_path:str=None,
                    img_dir:str=None,
                    img_suffix:str='.tif', 
                    rgb_label_np:np.ndarray=None, 
                    vis_save_path:str=None, 
                    vis_save_suffix:str=None,
                    vis_add_geo:bool=False, 
                    geo_img_path:str=None, 
                    geo_img_suffix:str='.tif'):
    
    file_name, mask_label_suffix = os.path.splitext(mask_label_path)
    file_name = file_name.replace('_18label', '')
    img_base_name = os.path.basename(file_name)
    img_path = os.path.join(img_dir, img_base_name+img_suffix)
    img_ds = gdal.Open(os.path.join(img_path))
    img_np = img_ds.ReadAsArray().transpose(1, 2, 0)  #(594,594,4)  # 这里是366附近吧
    if img_np.shape[2] == 3:   # RGB
        img_R = img_np[:,:,0]
        img_G = img_np[:,:,1]
        img_B = img_np[:,:,2]
    elif img_np.shape[2] == 4: # MSI 4 chan
        img_R = img_np[:,:,2]
        img_G = img_np[:,:,1]
        img_B = img_np[:,:,0]
    elif img_np.shape[2] == 10: # MSI 10 chan
        img_R = img_np[:,:,2]
        img_G = img_np[:,:,1]
        img_B = img_np[:,:,0]

    
    img_type = img_np.dtype
    nodata_pos = np.all((img_R == 0, img_G == 0, img_B == 0), axis=0)
    # print(nodata_pos)


    img_R[nodata_pos] = 0
    img_G[nodata_pos] = 0
    img_B[nodata_pos] = 0

    min_R, max_R = np.percentile(img_R, (2, 98))  # 去掉极端值
    min_G, max_G = np.percentile(img_G, (2, 98))  # 去掉极端值
    min_B, max_B = np.percentile(img_B, (2, 98))  # 去掉极端值

    # 归一化到 0-255
    img_R = np.clip((img_R - min_R) / (max_R - min_R) * 255, 0, 255).astype(np.uint8)
    img_G = np.clip((img_G - min_G) / (max_G - min_G) * 255, 0, 255).astype(np.uint8)
    img_B = np.clip((img_B - min_B) / (max_B - min_B) * 255, 0, 255).astype(np.uint8)  
    


    img_R = img_R.astype(np.uint8)
    img_G = img_G.astype(np.uint8)
    img_B = img_B.astype(np.uint8)
    
    # 三个通道单独均衡化
    # img_R = cv2.equalizeHist(img_R)
    # img_G = cv2.equalizeHist(img_G)
    # img_B = cv2.equalizeHist(img_B)

    # 只对大于0的地方进行均值化
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # 对每个通道应用 CLAHE
    img_R = clahe.apply(img_R)
    img_G = clahe.apply(img_G)
    img_B = clahe.apply(img_B)
    
    img_R[nodata_pos] = 0
    img_G[nodata_pos] = 0
    img_B[nodata_pos] = 0

    vis_img_np = np.stack([img_R, img_G, img_B], axis=2)
    

    # label_ds = gdal.Open(os.path.join(mask_label_path))
    # label_np = label_ds.ReadAsArray().transpose(1, 2, 0)  #(594,594,4)  # 这里是366附近吧
    # 将 nodata_pos 对应位置的 label_np 设为 [0, 0, 0]
    

    if vis_img_np.shape != rgb_label_np.shape:
        rgb_label_np = cv2.resize(rgb_label_np, (vis_img_np.shape[1], vis_img_np.shape[0]), interpolation=cv2.INTER_NEAREST)

    rgb_label_np[nodata_pos] = [0, 0, 0]
    # 权重越大，透明度越低
    # alpha = 0.3
    # overlapping = (vis_img_np * (1 - alpha) + rgb_label_np * alpha).astype(np.uint8)

    # 对vis_img_np 进行彩色的直方图均衡化
    overlapping = cv2.addWeighted(vis_img_np, 0.5, rgb_label_np, 0.5, 0)
    # overlapping = cv2.addWeighted(vis_img_np, 1.0, rgb_label_np, 0.0, 0)

    # overlapping = cv2.cvtColor(overlapping, cv2.COLOR_BGR2RGB)
    if vis_add_geo:
        vis_save_suffix = '.tif'
        vis_out_path = os.path.join(vis_save_path,os.path.basename(file_name)+vis_save_suffix)
        assert geo_img_path is not None, "当需要添加地理信息时, 请提供地理信息图像路径 `geo_img_path`!!!"
        geo_img_path_ = os.path.join(geo_img_path, os.path.basename(file_name)+geo_img_suffix)
        geo_img_ds = gdal.Open(geo_img_path_, gdal.GA_ReadOnly)
        assert  geo_img_ds is not None, f"无法打开 {geo_img_path_}, 请检查路径及后缀"
        trans = geo_img_ds.GetGeoTransform()
        proj = geo_img_ds.GetProjection()

        save_array_to_tif(img_array=overlapping,
                          out_path=vis_out_path,
                          Transform=trans,
                          Projection=proj,
                          Band=3,
                          Datatype=gdal.GDT_Byte)
    else:
        vis_save_suffix = '.png'
        vis_out_path = os.path.join(vis_save_path,os.path.basename(file_name)+vis_save_suffix)
        overlapping = cv2.cvtColor(overlapping, cv2.COLOR_BGR2RGB)
        cv2.imwrite(vis_out_path, overlapping)
    # import pdb; pdb.set_trace()


if __name__ == '__main__':

    MSI_img_dir = '../../2-多领域地物覆盖基础/GF2_5B-24类/6-用来出图的裁切小图/img_dir/val/'
    mask_label_dir = '../GF2的mask-19类/convnext-B-baseline-78k/'
    rgb_label_save_path = '../GF2-叠加底图和RGB标签'
    vis_save_path =  '../GF2-叠加底图和RGB标签'

    mask_suffix = '.png'

    mkdir_or_exist(vis_save_path)
    mkdir_or_exist(rgb_label_save_path)

    label_list = find_data_list(mask_label_dir, suffix=mask_suffix) 


    for mask_label_path in tqdm(label_list, colour='Green'):
        RGB_label_np = Mask2rgb_Without_Geo(mask_label_path=mask_label_path, 
                                            METAINFO=METAINFO, 
                                            reduce_zero_label=True, 
                                            save_rgb_label=False, 
                                            save_rgb_path=rgb_label_save_path,
                                            add_geo=True,
                                            geo_img_path=MSI_img_dir)
        # print(RGB_label_np.shape)
        Overlay_picture(mask_label_path=mask_label_path,
                        img_dir=MSI_img_dir,
                        rgb_label_np=RGB_label_np,
                        vis_save_path=vis_save_path,
                        vis_add_geo=True,
                        geo_img_path=MSI_img_dir)
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()

    atl_logger.info("Done!!!")
    # import pdb; pdb.set_trace()
