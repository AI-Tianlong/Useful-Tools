# Docker的命令
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
  docker run --gpus all --shm-size=40g -it --name atl-mmseg  8a2cd2aba91f
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
  docker exec -it 00b08fbae5cf /bin/bash
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
