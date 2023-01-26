## Jupyter相关的问题
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
