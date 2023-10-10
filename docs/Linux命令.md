## Linux常用命令  
### Download and unzip
```bash
wget http://dags.stanford.edu/data/iccv09Data.tar.gz -O stanford_background.tar.gz
tar xf stanford_background.tar.gz
```
### unzip  
```bash
unzip  -d 要解压缩到的文件夹路径 被解压的文件路径
```
### tar -xzvf
```bash
tar -xzvf xxxxxxxx.tar.gz
```
### Linux下查看文件的大小 (VSCODE中查看传输数据的多少)

```bash
watch -n 0.1 ls -lh
```
### 删除文件夹
```bash
sudo rm -rf xxx
```
### 删除文件
```none
rm的选项有：
-f, --force 强制删除，不需要确认
-i 每删除一个文件或进入一个子目录都要求确认
-I 在删除超过三个文件或者递归删除前要求确认
-r, -R 递归删除子目录
-d, --dir 删除空目录
-v, --verbose 显示删除结果
注意事项：需要谨慎使用rm删除命令，如果文件被删除，那么就不可以被恢复，若是没有使用- r选项，则rm命令不会删除目录。
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
如果很多的话，查看其父进程
```bash
ps -ef|grep PID
```
然后通过`kill -9 {PPID}` 来杀死进程
```bash
kill -9 xxx
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
