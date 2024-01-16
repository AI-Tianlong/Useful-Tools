## Git相关的命令


### 生成Github秘钥
[生成github秘钥](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
```bash
ssh-keygen -t ed25519 -C "839290771@qq.com"
```
### 添加秘钥到ssh-agent
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_
```
### 如果是第一次用git 设置用户信息
```bash
git config --global user.name "AI-Tianlong"
git config --global user.email  "xxxxxxxx@qq.com"
```
### docker的命令
- 帮助命令
  ```bash
  docker version # 显示版本信息
  docker info    # 显示一些更详细的信息，系统信息
  docker 命令 --help
  ```
- docker 下载镜像
  ```bash
  docker pull mmseg:[版本，不写默认最新]
  
- 镜像命令
  ```bash
  docker images # 查看所有镜像
  -a #列出所有的
  -f
  -q #只显示ID
  ```
- 创建容器，有了镜像才能创建容器实例
  ```bash
  docker run [可选参数] image
  --name="ATL1" "ATL2"# 用来区分容器
  -d 后台方式运行
  -it 使用交互方式运行，进入容器查看内容
  -p（小写）指定容器的端口 -p 主机端口:容器端口 可和主机映射
                        -p 容器端口
                        -p ip:主机端口:容器端口
  -P（大写） 随即指定端口
  ```
- 交互式进入docker
  ```bash
  dpcker run -it ATL1 /bin/bash
  ==> exit （退出）从容器中返回主机
  ==> P+Q+ctrl 容器不停止退出
  ```
- 查看所有运行的容器
  ```bash
  docker ps
  docker ps -a # 曾运行的+现在的
  ```
- docker 删除镜像
  ```bash
  docker rmi -f ID # 删除指定的容器，不能产出运行ing
  docker rmi -f $(docker images -aq) # 删除所有的容器,-f强制删除
  ```
- 启动和停止容器
  ```bash
  docker start 容器ID
  docker restart 容器ID
  docker stop 容器ID
  docker kill 容器ID # 暴力杀死
  ```
### git clone 一个仓库

```bash
git clone xxxxx --recursive
# clone 指定的分支
git clone -b dev-1.x https://github.com/open-mmlab/mmsegmentation.git
# git 完之后cd xxx
git submodule update --init
```
### git clone 代理
```bash
git clone https://ghproxy.com/xxxxxxxxx
```

### 用代理clone后修改url
```bash
git remote --set-urls -add xxxxx
```


### 添加上游仓库

```bash
git remote add upstream xxxxxxx
```

## 添加pre-commit

```bash
pip install -U pre-commit
pre-commit install
pre-commit run --all-files
```

### 切换至dev 1.x分支

```bash
git checkout dev-1.x 
```

### 创建分支

```bash
git checkout -b AI-Tianlong/support_Mapillary_dataset
```

### git add+commit+push
```bash
git add .
git commit -m "xxxxx"       (--no-verify,有pre-commit hook的话)
git push                    (第一次 git push -u origin 分支名字)
```
### git 添加代理
```bash
# socks
git config --global http.proxy 'socks5://127.0.0.1:10808' 
git config --global https.proxy 'socks5://127.0.0.1:10808'
# http
git config --global http.proxy http://127.0.0.1:10809 
git config --global https.proxy https://127.0.0.1:10809

# 只对github.com使用代理，其他仓库不走代理
git config --global http.https://github.com.proxy socks5://127.0.0.1:10808
git config --global https.https://github.com.proxy socks5://127.0.0.1:10808
# 取消github代理
git config --global --unset http.https://github.com.proxy
git config --global --unset https.https://github.com.proxy

```


### 连接github有问题？
C:\Windows\System32\drivers\etc  在hosts里加
```none
199.232.68.133 raw.githubusercontent.com
```
