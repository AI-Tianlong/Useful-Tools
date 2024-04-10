# 发布自己的库到pypi
想把自己的库发布到 pypi，需要首先在 [pypi](https://pypi.org/) 创建一个账号。
## 1 打包和发布的工具安装
在虚拟环境内安装`setuptools ` `build` `twine`三个库。  
其中，`setuptools `和`build`是将自己写的python代码进行打包，`twine`用来将自己的包发布带pypi。
```bash
pip install setuptools build twine
```
## 2 准备打包的文件
按照图示准备
![image](https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/7ff95674-f34d-46ad-b91e-f064e056767b)
### 2.1 准备 pyproject.toml
```bash
  [build-system]
    requires = ["setuptools", "setuptools-scm"]
    build-backend = "setuptools.build_meta"
    
  [project]   

    name = "ATL_Tools"
    version = "1.0.6"
    description = "AI-Tianlong的Tools打包，开箱即用, 包含ATL_path和ATL_gdal,可用于遥感图像处理"
    readme = "README.md"
    keywords= ["ATL_Tools", "GDAL", "AI-Tianlong", "Chinese"]
    license = {text = "MIT License"}

```
## 3 打包
在`pyproject.toml`文件所在的目录下执行以下命令
```bash
python -m build
```
成功执行结束后，会在命令行中提示<font color='green'>`Successfully built ATL_Tools-1.0.6.tar.gz and ATL_Tools-1.0.6-py3-none-any.whl`</font>
## 4 发布到pypi
在打包的目录执行以下目录，输入pypi的API即可完成上传
```bash
twine upload dist/*
```
## 5 在pypi查看
https://pypi.org/project/ATL-Tools/
   
