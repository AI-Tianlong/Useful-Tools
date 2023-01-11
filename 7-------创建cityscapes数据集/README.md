### 使用方式
* [下载Cityscapes](https://www.cityscapes-dataset.com/downloads/)数据集，下载后会存在两个压缩文件`gtFine_trainvaltest.zip`和`leftImg8bit_trainvaltest.zip`，分别为标签和图像。
* 假设您的两个压缩文件目录在`data/cityscapes`下，执行 `unzip gtFine_trainvaltest.zip`和`unzip leftImg8bit_trainvaltest.zip`命令分别对这两个压缩文件解压，会生成`gtFine`和 `leftImg8bit`两个文件夹如下图

    <div align=center><img src='https://github.com/AI-Tianlong/Useful-Tools/blob/main/Pictures/cityscapes_gtFine_before_convert.png' width=30% align="middle" ></div>

    可以看到此时的gtFine中，每一幅图像有四种类型的标签
    ![](https://github.com/AI-Tianlong/Useful-Tools/blob/main/Pictures/cityscapes_gtFine_before_convert_show.png)
* <font color='red'>将目录切换至该py文件下</font>，运行以下命令 [参考mmsegmentation cityscapes](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#cityscapes)
    ```bash
  # --nproc means 8 process for conversion, which could be omitted as well.
  python cityscapes.py data/cityscapes --nproc 8
    ```
 * 运行上述命令后，会在`gtFine`文件中生成`**labelTrainIds.png`,该图像即为训练所需的标签。







### 数据集下载地址

https://www.cityscapes-dataset.com/downloads/

### 数据集介绍

https://blog.csdn.net/lx_ros/article/details/125667076

### 官方的Github

https://github.com/mcordts/cityscapesScripts/blob/master/README.md
