# Useful-Tools
Some Useful Tools Code
本仓库存储了一些平时写的小Tools，不规范但能实现自己的idea。  
给自己备份以便将来之需。  

## 一些常用的命令
### unzip
```bash
unzip  -d 要解压缩到的文件夹路径 被解压的文件路径
```

### 给Jupyter notebook添加kernel

```bash
python -m ipykernel install --user --name xxx --display-name "Python (xxx)"
```
### 删除jupyter ipykernel
```bash
jupyter kernelspec remove xxx
```

### Linux下查看文件的大小 (VSCODE中查看传输数据的多少)

```bash
watch -n 0.1 ls -lh
```
### Git clone 代理

```bash
git clone https://ghproxy.com/xxxxxxxxx
```

### 创建符号链接
```bash
ln -s /HOME/scz5158/run/ATL/OpenMMLab/Dataset/cityscapes ./
```
### Chrome不能翻译
* C:\Windows\System32\drivers\etc 找到hosts
* 参考
* https://blog.csdn.net/w_p_wyd/article/details/121836304
* https://zhuanlan.zhihu.com/p/576290326
```bash
# 我找到了几个可用的谷歌国外 IP，不保证长期可用，后续也不保证更新，有能力的建议自己找。
# 任选一个加到 Hosts 文件中并重启浏览器，建议使用前先在 CMD 中 Ping 一下确保可用。

# 注意！添加以下内容时，请不要在开头加上 # 井号，# 是注释符，行首加了 # 就等于这行没写！

# 提示：添加以下内容之前，请先删除 Hosts 文件中以前添加过的所有 translate.googleapis.com 内容，避免因为顺序而被覆盖！

142.250.4.90 translate.googleapis.com
142.250.30.90 translate.googleapis.com
142.250.99.90 translate.googleapis.com
142.250.101.90 translate.googleapis.com
142.250.105.90 translate.googleapis.com
142.250.107.90 translate.googleapis.com
172.253.112.90 translate.googleapis.com
172.253.114.90 translate.googleapis.com
172.253.116.90 translate.googleapis.com
172.253.124.90 translate.googleapis.com

# 如果你有 IPv6 可以用下面的这些，没有请勿使用
2800:3f0:4004:806::200a translate.googleapis.com
2800:3f0:4004:805::200a translate.googleapis.com
2800:3f0:4003:c02::5f translate.googleapis.com

# 注意：这些 IP 只能指向谷歌翻译 API 接口域名，指向其他谷歌域名是无法使用的。
```
