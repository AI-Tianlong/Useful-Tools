#!/usr/bin/env bash


#第一步，转换训练集标签为18类 get
python yuchuli.py     

#第二步，开始训练 get
bash train.sh my_config/mask2former_beit_batch2_4w.py 

#第三步，权重平均 get
python weights_average.py

#第四步，测试集预处理 get
python ceshijiyuchuli.py

#第五步，测试集推理
bash test.sh my_config/mask2former_beit_batch2_4w.py /home/xiaopengyou1/AITianlong/ViT-Adaptor/segmentation/work_dirs/mask2former_beit_batch2_4w/swa_4k_4w2.pth

#第六步 results标签转换  get
python houchuli.py

#第七步 results.zip   get
cd work_dirs/mask2former_beit_batch2_4w/
zip -r results.zip results/

#第八步 拷贝results.zip-->/output
cp -r results.zip /output
