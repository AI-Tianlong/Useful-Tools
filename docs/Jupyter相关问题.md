## Jupyter相关的问题
### conda 打包环境为tar.gz 并解压
```bash
pip install conda-pack
# conda pack -n 要打包的环境名称 -o 输出的tar.gz
conda pack -n openmmlab -o ATL-track.tar.gz --ignore-editable-packages #忽略`pip install -e .`安装的包

mkdir ~/ananconda/envs/ATL-track
tar -xf pcdet.tar.gz -C ~/ananconda/envs/ATL-track
```
### 给Jupyter notebook添加kernel

```bash
python -m ipykernel install --user --name xxx --display-name "Python (xxx)"
```
### 删除jupyter ipykernel
```bash
jupyter kernelspec remove xxx
```
### conda 创建环境ATL
```bash
conda create -n ATL python=3.9
```
### conda删除环境
```bash
conda rm -n ATL --all
```
### conda修改envs_dirs
在 C/user/.condarc 下，修改envs_dirs:
```bash
envs_dirs:
  - D:\Application\ATL_APP\miniconda\miniconda\envs
channels:
  - defaults
show_channel_urls: true

```
要是还不行，那就偷懒
```bash
conda create -p D:\Application\ATL_APP\miniconda\miniconda\envs\pytorch python=3.9
```
