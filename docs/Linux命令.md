## Linux常用命令  
### unzip  
```bash
unzip  -d 要解压缩到的文件夹路径 被解压的文件路径
```
### Linux下查看文件的大小 (VSCODE中查看传输数据的多少)

```bash
watch -n 0.1 ls -lh
```
### 创建符号链接
```bash
ln -s /HOME/scz5158/run/ATL/OpenMMLab/Dataset/cityscapes ./
```
### 查看显存占用
```bash
watch -n 0.1 nvidia-smi
```
### 查看显存进程
```bash
fuser -v /dev/nvidia*
```
### terminal忽略大小写补全
编辑 vim ~/.inputrc 
文件设置 (实测Ubuntu14是   /etc/.inputrc   文件)
文件末尾添加如下代码:  
```
vim ~/.inputrc
```
```bash
# do not show hidden files in the list
set match-hidden-files off
 
# auto complete ignoring case
set show-all-if-ambiguous on
set completion-ignore-case on

"\e[A": history-search-backward
"\e[B": history-search-forward
```
### VSCODE打开图片提示视图错误
```bash
#linux
rm -rf ~/.config/Code/Cache

#windows
Go to the file explorer and to the path 
C:\Users\<user_name>\AppData\Roaming\Code 
and clear the contents of the folders Cache, 
CachedData, CachedExtensions, CachedExtensionVSIXs 
(if this folder exists) and Code Cache.

#macos
~/Library/Application Support/Code/
```


参考https://blog.csdn.net/qq_40309341/article/details/121354666
### ubuntu查看内存使用情况
```bash
free -mh 
```
### ubuntu查看硬盘使用情况
```bash
df -lh
```
