# Docker的命令
- 新建一个用户，添加至 docker 组
  [【官方文档】](https://docs.docker.com/engine/install/linux-postinstall/)
  - 添加一个 `docker` 组
    ```bash
    sudo groupadd docker
    ```
  - 添加用户到 `docker` group.
     ```bash
     sudo usermod -aG docker $USER
     ```
  - 验证
    ```bash
    docker run hello-world
    ``` 
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
- 重命名镜像
  ```bash
  docker tag oldname:oldtag newname:newtag
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
- Docerfile 创建镜像
  ```bash
  # 切换到Dockerfile所在的路径
  docker build -t {镜像名字}:{tag} ./
  ```
  ```bash
  # 创建容器
  docker run --gpus all --shm-size=50g -it -v /opt/AI-Tianlong/2024bisai-docker/2024-ISPRS/water/input_path:/input_path -v /opt/AI-Tianlong/2024bisai-docker/2024-ISPRS/water/output_path:/output_path atl-mmseg-water:v1
  ```
  ```bash
  # 如果容器有启动命令,加上/bin/bash
  docker run --gpus all --shm-size=50g -it -v /opt/AI-Tianlong/2024bisai-docker/2024-ISPRS/water/input_path:/input_path -v /opt/AI-Tianlong/2024bisai-docker/2024-ISPRS/water/output_path:/output_path registry.cn-hangzhou.aliyuncs.com/mask2former/atl-mmseg:cuda-11.6-v11
  ```
  ```bash
  # 往容器里写入CMD并commit为镜像
  docker commit -c 'CMD ["python", "run.py","/input_path","/output_path"]' <container_id_or_name> <new_image_name>
  ```
  ```Dockerfile
  FROM atl-mmseg-water:latest
  
  ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
  ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
  ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
  
  # To fix GPG key error when running apt-get update
  RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
  RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
  
  RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-dev  \
      && apt-get clean \
      && rm -rf /var/lib/apt/lists/*
  
  COPY . /workspace
  WORKDIR /workspace
  CMD ["/bin/bash", "run.sh"]
  ```
- 交互式进入docker
  在最后加 `/bin/bash`即可
  ```bash
  dpcker run -it ATL1 /bin/bash
  docker exec -it 00b08fbae5cf /bin/bash
  docker exec -it atl_cambricon_pytorch113_ubuntu2004 /bin/bash
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
