## 使用方式
该程序的目标是生成如下格式的cityscapes数据集  
```txt
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
|   |   |   |   ├── aachen_000000_000019_leftImg8bit.png
|   |   |   |   ├── aachen_000001_000019_leftImg8bit.png
│   │   │   ├── val
|   |   |   |   ├── frankfurt_000000_000294_leftImg8bit.png
|   |   |   |   ├── frankfurt_000000_000576_leftImg8bit.png
│   │   ├── gtFine
│   │   │   ├── train
|   |   |   |   ├── aachen_000000_000019_gtFine_labelTrainIds.png
|   |   |   |   ├── aachen_000001_000019_gtFine_labelTrainIds.png
│   │   │   ├── val
|   |   |   |   ├── frankfurt_000000_000294_gtFine_labelTrainIds.png
|   |   |   |   ├── frankfurt_000000_000576_gtFine_labelTrainIds.png
```
* [下载Cityscapes](https://www.cityscapes-dataset.com/downloads/)数据集，下载后会存在两个压缩文件`gtFine_trainvaltest.zip`和`leftImg8bit_trainvaltest.zip`，分别为标签和图像。
* 假设您的两个压缩文件目录在`data/cityscapes_zip`下，执行 `unzip gtFine_trainvaltest.zip`和`unzip leftImg8bit_trainvaltest.zip`命令分别对这两个压缩文件解压，会生成`gtFine`和 `leftImg8bit`两个文件夹如下图

    <div align=center><img src='https://github.com/AI-Tianlong/Useful-Tools/blob/main/Pictures/cityscapes_gtFine_before_convert.png' width=30%></div>

    可以看到此时的gtFine中，每一幅图像有四种类型的标签
    ![](https://github.com/AI-Tianlong/Useful-Tools/blob/main/Pictures/cityscapes_gtFine_before_convert_show.png)
* <font color='red'>将目录切换至该py文件下</font>，运行以下命令 [参考mmsegmentation cityscapes](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#cityscapes)
    ```bash
  # --nproc means 8 process for conversion, which could be omitted as well.
  python cityscapes.py data/cityscapes_zip --nproc 8
  ```
 * 运行上述命令后，会在`gtFine`文件中生成`**labelTrainIds.png`，该图像即为训练所需的标签。以及三个文件`train.txt`、`test.txt`、`val.txt`，存放着train、test、val数据集的图像名称，接下来需要用该名称以及获得的`**labelTrainIds.png`去生成最终的究极无敌版数据集！
    如下所示
   <img src='https://github.com/AI-Tianlong/Useful-Tools/blob/main/Pictures/cityscapes_gtFine_after_convert.png' width=30%><img src='https://github.com/AI-Tianlong/Useful-Tools/blob/main/Pictures/cityscaprs_txt_file.png'><img src='https://github.com/AI-Tianlong/Useful-Tools/blob/main/Pictures/cityscapes_txt_conent.png'>

    ![](https://github.com/AI-Tianlong/Useful-Tools/blob/main/Pictures/cityscapes_gtFine_after_convert_show.png)
    其中，像素值为255的是需要忽略的标注。其余像素值已经被转换为所需要的19类类别，具体细节可以看[官方GitHub](https://github.com/mcordts/cityscapesScripts/blob/master/README.md)的。
* 生成完整的cityscapes数据集
  ```bash
  python Create_CityScapes_use_txt_multi_progress.py  --out-dir [Your out dir]
  ```



## 数据集下载地址

https://www.cityscapes-dataset.com/downloads/

## 数据集介绍

https://blog.csdn.net/lx_ros/article/details/125667076

## 官方的Github

https://github.com/mcordts/cityscapesScripts/blob/master/README.md
