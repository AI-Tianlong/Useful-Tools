## 给VSCODE 添加 Gitbash 终端
`ctrl + shift + P`打开VSCODE设置（json）  
添加以下代码：
```txt
"terminal.integrated.profiles.windows": {
        "PowerShell": {
            "source": "PowerShell",
            "icon": "terminal-powershell"
        },
        "Command Prompt": {
            "path": [
                "${env:windir}\\Sysnative\\cmd.exe",
                "${env:windir}\\System32\\cmd.exe"
            ],
            "args": [],
            "icon": "terminal-cmd"
        },
        "Git Bash": {
            "path": "D:/Applications/Git/Git/bin/bash.exe",
            "args": []
        }
    },
    "terminal.integrated.defaultProfile.windows": "Git Bash"
```
## VSCODE中的Gitbash 自动激活 conda环境
```bash
vim ~/.bashrc
```
```
source D:/Applications/miniconda/miniconda/Scripts/activate openmmlab
```
## Jupyter相关的问题
### pip阿里镜像源
```bash
-i https://mirrors.aliyun.com/pypi/simple
```

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
### conda 激活环境
```bash
conda activate ATL
```
### conda deactivate 环境
```bash
conda deactivate ATL
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
