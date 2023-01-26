"""
将24位的真彩图像转换为8位的彩图

"""
from PIL import Image
import numpy as np

img_path = r'C:\Users\w8392\Desktop\test.jpg'
save_path = r'C:\Users\w8392\Desktop\test_after.jpg'


img = Image.open(img_path).convert('P') # .convert('L')是8bit的黑白图
img.save(save_path) # 转换后的进行保存,需要PNG格式，JPEG P有问题