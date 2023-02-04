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
