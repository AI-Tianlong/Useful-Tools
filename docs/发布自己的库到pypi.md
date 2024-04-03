# 发布自己的库到pypi
想把自己的库发布到 pypi，需要首先在 [pypi](https://pypi.org/) 创建一个账号。
1. 在虚拟环境内安装`setuptools ` `build` `twine`三个库。  
   其中，`setuptools `和`build`是将自己写的python代码进行打包，`twine`用来将自己的包发布带pypi。
   ```bash
  pip install setuptools build twine
   ```
