import ee  #导入ee `pip install earthengine-api`
import geemap
from IPython.display import Image
import datetime
import os

# 使用 `pip install ATL-Tools` 安装
from ATL_Tools import mkdir_or_exist, find_data_list 
from tqdm import tqdm

print("正在验证身份。。。")
# 身份验证，会跳转到浏览器进行验证
ee.Authenticate() 

#初始化项目，编号在GEE查看
ee.Initialize(project='applied-tractor-343704')  
print("验证通过")


sentinel2_images = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')



# shp 转换为 ee 格式
# Harbin_shp = "./nangangqu/nangangqu.shp"
# Harbin_ee = geemap.shp_to_ee(Harbin_shp)  
json_files_path = "E:/Datasets/ATL_ATL自建数据集/5billion-sentinel2/名字命名的矢量json"
json_files = find_data_list(json_files_path, "json")


flag_name = None  # 用来检测下在哪一个的时候报错

start_time = "2019-07-1"
end_time = "2019-08-15"

s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

# Cloud Score+ image collection. Note Cloud Score+ is produced from Sentinel-2
# Level 1C data and can be applied to either L1C or L2A collections.
csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')

# Use 'cs' or 'cs_cdf', depending on your use-case; see docs for guidance.
QA_BAND = 'cs_cdf'

# The threshold for masking; values between 0.50 and 0.65 generally work well.
# Higher values will remove thin clouds, haze & cirrus shadows.
CLEAR_THRESHOLD = 0.60

def func_hfm(img):
    return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD)).divide(10000)



def download_function(flag_name):
    download_path = "E:/Datasets/ATL_ATL自建数据集/5billion-sentinel2/sentinel-2-images"
    mkdir_or_exist(download_path)
    
    for geojson in tqdm(json_files, desc="正在下载数据："):      
        # print(geojson)
        if "\\" in geojson:
            geojson = geojson.replace("\\", "/")
        # print(geojson)

        have_download_file = find_data_list(download_path, ".tif")
        have_download_img = [os.path.basename(i) for i in have_download_file]
        
        # 是否下载的标志位, 默认为True
        download_flag = True    
        geojson_basename = os.path.basename(geojson).split('.')[0]

        for img_name in have_download_img:
            if geojson_basename in img_name:
                print(f"当前处理的json文件为：{geojson} 已经下载过了，跳过")
                download_flag = False
                break

        if download_flag:       

            #每次下载都给他验证一次
            # 身份验证，会跳转到浏览器进行验证
            ee.Authenticate() 
            #初始化项目，编号在GEE查看
            ee.Initialize(project='applied-tractor-343704')  

            ## json 转换为 ee 格式 （从最开始下载的json）
            geemap_ee = geemap.geojson_to_ee(geojson) 
            print(f"当前处理的json文件为：{geojson}") 


            # ROI 区域为 矢量边界最小的外接矩形
            ROI = geemap_ee.geometry().bounds()

            # 时间段 2022 年全年、云层覆盖率小于 10%、以目标区域为边界、遍历去云函数、选取 1~8 波段
            collection = s2.filterBounds(geemap_ee) \
                .filterDate(start_time, end_time) \
                .linkCollection(csPlus, [QA_BAND]) \
                .map(func_hfm) \
                .select('B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12')

            # collection = sentinel2_images.filterDate('2019-6-10', '2019-10-1') \
            # .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5)) \
            # .filterBounds(geemap_ee) \
            # .map(maskS2clouds) \
            # .select('B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12')


            # 将影像数据集计算中值后得到的单幅影像针对目标区域进行裁剪，得到最终待下载数据
            composite = collection.median().clip(geemap_ee) #ROI

            #查看影像

            # 可视化 注释掉
            # Map.addLayer(composite, rgbVis, '南岗区S2影像')
            # Map.addLayer(geemap_ee.style(**styling), {}, "南岗区边界")  # 将矢量数据添加到地图上
            # Map.centerObject(geemap_ee, 9)

            # 配置输出目录
            output_file = os.path.join(download_path, 'images') 
            
            print(f'文件将要下载到{output_file}')

            mkdir_or_exist(output_file)

            out_tif = os.path.join(output_file, 
                                   'S2_SR_2019_'+ 
                                   os.path.basename(geojson).split('.')[0]+'.tif') # GF2_PMS1__L1A0000564539-MSS1
            
            flag_name = out_tif # 查看下载错误的时候是哪一个报错

            # 下载影像
            geemap.download_ee_image(
            image=composite,
            filename=out_tif,
            region=ROI,
            crs="EPSG:4326",
            scale=10,
            )

        


if __name__ == "__main__":
    # 设置尝试执行的最大次数
    print("开始下载数据...")
    max_attempts = 200
    attempt = 0
    flag_name = ''
    error_list = []

    while attempt < max_attempts:
        try:
            # 在此处编写您的代码，例如：
            download_function(flag_name)  # 这里是一个会导致除零错误的示例代码，您可以替换为您自己的代码
            # 如果没有错误，则跳出循环
            break
        except Exception as e:
            attempt += 1
            error_list.append(flag_name)
            print(f"Attempt {attempt} failed with error: {e}")
            with open(f"下载错误的列表{attempt}.txt", "w") as f:
                f.write(str(error_list))

    # 如果已经尝试了最大次数仍然失败，则输出错误消息
    if attempt == max_attempts:
        print(f"最大次数已达到，但是仍然失败了 \n 下载错误的列表：{error_list}")
