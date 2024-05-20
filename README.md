# Useful-Tools
Some Useful Tools Code
本仓库存储了一些平时写的小Tools，不规范但能实现自己的idea。  
给自己备份以便将来之需。  

## 一些常用的命令
* [Linux命令](docs/Linux命令.md)
* [markdown语法]()
* [Git相关的](docs/Git相关.md)
* [Docker相关的](docs/docker相关的.md)
* [那些从chatGPT中学来的算法实现](/docs/%E9%82%A3%E4%BA%9B%E4%BB%8EchatGPT%E4%B8%AD%E5%AD%A6%E6%9D%A5%E7%9A%84%E7%AE%97%E6%B3%95%E5%AE%9E%E7%8E%B0.md)
* [给自己粘贴图床](docs/%E7%BB%99%E8%87%AA%E5%B7%B1%E7%B2%98%E8%B4%B4%E5%9B%BE%E5%BA%8A.md)
* [发布自己的库到pypi](docs/发布自己的库到pypi.md)
* [QGIS 使用技巧相关](docs/QGIS使用相关的.md)
* [Jetson AGX Orin 命令](docs/Jetson%20AGX%20Orin命令.md)
* [ATL_Path.py](code/0-------ATL_path.py)
* [Conda相关的](docs/Conda相关问题.md)
* [超算相关的](docs/超算相关的.md)
* [数据集相关的](docs//数据集相关的.md)
* [好玩的大模型](docs/好玩的大模型.md)
* [文章的Note](docs/文章的Note.md)
* [Chrome翻译问题](docs/Chrome翻译问题.md)
  
## 一些超好用的宝藏软件
* 截图软件[Snipaste](https://www.snipaste.com/)
* 图片查看[nomacs](https://nomacs.org/)
* 录屏[OBS](https://obsproject.com/)
* 画流程图[draw.io](https://draw.io)
* 文本编辑[notepad++](https://notepad-plus-plus.org/downloads/)
* GitHub查看代码tree octotree
* [视频格式转换Movavi Video Converter]

## VSCODE好用的插件
- 自动补全文件夹路径 [Path Autocomplete](https://marketplace.visualstudio.com/items?itemName=ionutvmi.path-autocomplete)
- [Git History](https://marketplace.visualstudio.com/items?itemName=donjayamanne.githistory)
- Docker

## 一些code
### 数据集相关的
#### ATL超级推荐的
> * [RGB2mask.py](https://github.com/AI-Tianlong/Useful-Tools/blob/main/code/0-------RGB2mask.py) 使用 mmseg的palette转换RGB为mask
> * [mask2RGB.py](https://github.com/AI-Tianlong/Useful-Tools/blob/main/code/0-------mask2RGB.py) 使用 mmseg的palette创建可视化RGB
> * [裁切大图为小图进行推理](https://github.com/AI-Tianlong/Useful-Tools/blob/main/code/0-------crop_patch_to_inference.py) 使用gdal裁切大图为小图
> * [合并推理后的小mask为大mask](https://github.com/AI-Tianlong/Useful-Tools/blob/main/code/0-------hebing_inference_to_big_mask.py) 使用gdal合并小的推理结果为大图
> * [mask2RGB并添加坐标信息](https://github.com/AI-Tianlong/Useful-Tools/blob/main/code/0-------mask2RGB_add_meta.py) 使用gdal转化mask为RGB并添加坐标信息
> * [合并24标签为12标签](https://github.com/AI-Tianlong/Useful-Tools/blob/main/code/0-------convert_24classes_to12.py) 使用gdal转化mask为RGB并添加坐标信息
* [创建cityscapes数据集](code/7-------%E5%88%9B%E5%BB%BAcityscapes%E6%95%B0%E6%8D%AE%E9%9B%86/)
* [创建mapillary数据集](code/8-------%E5%88%9B%E5%BB%BAmapillary%E6%95%B0%E6%8D%AE%E9%9B%86/)
* GID数据集
  * [RGB2mask](code/17-----GID_create_masks_png.ipynb)
  * [crop](code/19-----GID_crop_images.ipynb)
* LoveDA数据集
  * [mask2RGB](code/20------LoveDA_create_vis_png.ipynb)
  * [crop](code/21------LoveDA_crop_images.ipynb)
* Potsdam & Vaihingen数据集
  * [RGB2mask](code/23------Vaihingen_Potsdam_create_masks_png.ipynb)
  * [crop](code/25------Vaihingen_Potsdam_crop_images.ipynb)
* 扫描文件夹中的path
  * [Path](code/path.py)
### 一些小tools
* [1 计算两张图像的CCA(多进程)](code/2.5-----%E8%AE%A1%E7%AE%97%E5%9B%BE%E5%83%8F%E7%9A%84CCA%E5%A4%9A%E8%BF%9B%E7%A8%8B.py)
* [2 根据CCA的计算结果生成数据集](code/3-------%E6%A0%B9%E6%8D%AECCA%E7%BB%93%E6%9E%9CCreate_Dataset.py)
* [3 空洞填充去孔后处理](code/5-------%E7%A9%BA%E6%B4%9E%E5%A1%AB%E5%85%85%E5%8E%BB%E9%99%A4%E7%A9%BA%E9%9A%99.py) 
* [4 Augumentor库增强后rename](https://github.com/AI-Tianlong/Useful-Tools/blob/main/code/6-------Augumentor_rename.py)
* [5 24bit_to_8bit](https://github.com/AI-Tianlong/Useful-Tools/blob/main/code/8-------24bit_2_8bit.py)
* [6 训练相关的](https://github.com/AI-Tianlong/Useful-Tools/blob/main/code/26------train_final.py)
* [7 计算数据集的标准差](https://github.com/AI-Tianlong/Useful-Tools/blob/main/code/14------%E8%AE%A1%E7%AE%97%E6%95%B0%E6%8D%AE%E9%9B%86%E7%9A%84%E6%A0%87%E5%87%86%E5%B7%AE.ipynb)
* [8 查看模型](https://github.com/AI-Tianlong/Useful-Tools/blob/main/code/10------%E6%9F%A5%E7%9C%8B%E6%A8%A1%E5%9E%8B.ipynb)
* [ATL_path](https://github.com/AI-Tianlong/Useful-Tools/blob/main/code/ATL_path.py)
### ⭐ state  
<img src='https://starchart.cc/AI-Tianlong/Useful-Tools.svg' width='70%'/>
