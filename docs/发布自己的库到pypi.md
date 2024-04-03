# 发布自己的库到pypi
想把自己的库发布到 pypi，需要首先在 [pypi](https://pypi.org/) 创建一个账号。
## 1 打包和发布的工具安装
在虚拟环境内安装`setuptools ` `build` `twine`三个库。  
其中，`setuptools `和`build`是将自己写的python代码进行打包，`twine`用来将自己的包发布带pypi。
```bash
pip install setuptools build twine
```
## 2 准备打包的文件
## 3 打包
在`pyproject.toml`文件所在的目录下执行以下命令
```bash
python -m build
```
成功执行结束后，会在命令行中提示<font color='green'>`Successfully built ATL_Tools-1.0.6.tar.gz and ATL_Tools-1.0.6-py3-none-any.whl`</font>
## 4 发布到pypi
```bash
twine upload dist/*
```
   
