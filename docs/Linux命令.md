## Linux常用命令 
## 关机命令
```bash
sudo shutdown     #一分钟内广播，清空缓存，然后一分钟时间一到就关机
```
```bsah
sudo shutdown -h now    #shutdown -h now 是关闭运行的程序后，刷新缓冲区后再调用init 0关机
```
```bsah
sudo init 0     #这个是直接关机，不管系统内运行的程序
```
## 查看系统信息
```bash
lsb_release -a
```
# 查看cpu的信息
```bash
cat /proc/cpuinfo
lscpu
cat /proc/cpuinfo | grep "model name"
```
## 停止桌面服务
```bash
service gdm3 stop
service gdm3 start
```
## 记录终端的输出
```bash
# 开始记录
script output_file.log
# 结束保存
script end

nvidia-smi > atl.log
```
## 寻找文件

![image](https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/8ccbe8b0-171f-4eca-8cf9-236d1c1d2b30)


## 两台服务器传输文件 这速率老快了100MB/s
端口号 -P 要放在前面，不然提示没有权限
1. 从服务器上下载文件
```bash
#scp -P 端口号 user@ip地址:{远程目录} {本地目录}
scp -P 60001 xxxx@xxxx:AI-Tianlong/Datasets/cityscapes/leftImg8bit_trainvaltest.zip ./
```
2. 从服务器上下载文件夹
```bash
#scp -r -P 端口号  user@ip地址:{远程目录} {本地目录}
scp -r -P 60001  xxxx@xxxx:AI-Tianlong/Datasets/cityscapes/ ./
```
3. 上传文件到服务器
```bash
#scp -P 端口号 {本地目录} user@ip地址:{远程目录} 
scp -P 60001 ./ATL.zip  xxxx@xxxx:AI-Tianlong/Datasets/cityscapes/leftImg8bit_trainvaltest.zip 
```
4. 上传文件夹到服务器
```bash
#scp -P 端口号 -r {本地目录}  user@ip地址:{远程目录} 
scp -P 1530 -r  imagenet/ xxxx@xxxxxx:/opt/AI-Tianlong/Datasets/imagenet
```
## 查看当前文件夹的挂载点
```bash
df -h .
df -h "$(pwd)"
```
## 查看当前文件夹目录
```bash
pwd
```
## 查看文件夹的大小
```bash
du -hc

sudo apt install ncdu
ncdu 
```
### wget 和 curl 下载文件
#### wget
```bash
wget "https://msravcghub.blob.core.windows.net/simmim-release/swinv2/pretrain/swinv2_giant_22k_500k.pth?se=2049-12-31&sp=r&sv=2022-11-02&sr=b&sig=aVSY0TZymdDvMLWm4Os1neOIlKR28Herw6o4hz2TqpI%3D" -O swinv2_giant_22k_500k.pth
```
#### curl
```bash
curl -o swinv2_giant_22k_500k.pth "https://msravcghub.blob.core.windows.net/simmim-release/swinv2/pretrain/swinv2_giant_22k_500k.pth?se=2049-12-31&sp=r&sv=2022-11-02&sr=b&sig=aVSY0TZymdDvMLWm4Os1neOIlKR28Herw6o4hz2TqpI%3D"
```
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
要使用tar命令解压缩文件到一个新文件夹，可以使用以下命令：
```bash
mkdir new_folder
tar -xzvf archive.tar -C new_folder
```
这里，mkdir new_folder 创建一个名为new_folder的新文件夹，tar -xf archive.tar -C new_folder 将archive.tar文件解压到这个新文件夹中。
- -x 表示解压缩。
- -f 指定文件名。
- -C 指定解压缩到的目录。
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
### VSCODE通过秘钥 ssh 服务器
参考资料：
- https://zhuanlan.zhihu.com/p/497462191
- https://blog.csdn.net/m0_54706625/article/details/129721121
 步骤1：本地电脑生成SSH秘钥
- 在本地电脑生成秘钥
 ```bash
 ssh-keygen 
 ```
- 将本地生成的公钥`xxxx.pub`中的内容，上传到服务器/home/atl/.ssh中
  将`xxxx.pub`中的内容，写入`.ssh/authorized_keys`
  ```bash
  cat xxxx.pub >> authorized_keys
  ```
- 在`/etc/ssh/sshd_config`中编辑
  ```bash
  sudo vim /etc/ssh/sshd_config
  ```
  ```text
  RSAAuthentication yes  
  PubkeyAuthentication yes 
  ```
- 注意设置权限问题！！！
  ```bash
  sudo chmod 700 ~/.ssh/
  sudo chmod 700 /home/atl #这个尤其容易忽视掉，我就是从这个坑里爬出来。有木有很高兴呀！,只能700，
  sudo chmod 600 ~/.ssh/authorized_keys
  ```
### VSCODE 取消自动关闭文件
https://blog.csdn.net/xhtchina/article/details/109773806
### VSCODE 不显示代码颜色
![image](https://github.com/AI-Tianlong/Useful-Tools/assets/50650583/a0819a60-2dcc-44ab-b1d1-6ae5593e3267)

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
### nvcc 不好使
```bash
export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
### ubuntu分辨率不能调，只有1024*768
![image](https://github.com/user-attachments/assets/37148371-8644-4095-b7a2-52e6468d5312)
上述调完之后，如果能正常进入界面，极大地好事，如果不能。  
开机只显示一个`_`的话，则按 `Ctrl+Alt+F1~F7` 进入 tty 登录，然后删掉`/etc/X11/xorg.conf`  
```bash
sudo rm -f /etc/X11/xorg.conf
```
