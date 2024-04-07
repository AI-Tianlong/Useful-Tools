import os
import imageio
from ATL_Tools import mkdir_or_exist, find_data_list

jpg_path = r'D:/ATL/SIRS/ATL汇报PPT/汇报的图/20240318/偏差/'
save_gif_path = "./偏差2.gif"

jps_list  = find_data_list(jpg_path, suffix='.jpg') # 获取jpg_path下的所有jpg文件
frames = []


for img_path in jps_list: # 读取image下的图片名称
    frames.append(imageio.imread(img_path))

# loop = 0 意思是无限循环
imageio.mimsave(save_gif_path, frames, 'GIF',fps=1, loop=0) # 保存在当前文件夹

