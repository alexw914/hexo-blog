---
title: CUDA镜像
date: 2026-08-27 17:41:26
categories: 
  - "Linux&工具"
tags:
  - Linux
  - CUDA
  - Docker
---

# CUDA开发Docker环境配置

## 1. 镜像下载

首先需要下载CUDA镜像，可以从官方仓库或者第三方[镜像站](https://docker.aityp.com/r/docker.io/nvidia/cuda)下载。使用官方镜像的话需要先配置key，具体可以参考[官方文档](https://docs.nvidia.com/ngc/latest/ngc-catalog-user-guide.html#account-signup)。

注意这里开发的话需要选择devel版本，以便进行编译，并且最好附带cudnn。

```shell
docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
docker tag swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 docker.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
```

## 2. 制作新镜像

官方镜像一般只有一些基础功能，需要安装一些工具库以及编译工具，例如nodejs、python、cmake，并配置远程连接，这里使用Dockerfile在基础版本上制作一个新镜像, 并且把一些工具源换掉。

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/ \
    NPM_CONFIG_REGISTRY=https://registry.npmmirror.com

# Ubuntu官方源替换为阿里云源，并设置网络重试
RUN sed -i \
        -e 's@http://archive.ubuntu.com/ubuntu/@https://mirrors.aliyun.com/ubuntu/@g' \
        -e 's@http://security.ubuntu.com/ubuntu/@https://mirrors.aliyun.com/ubuntu/@g' \
        /etc/apt/sources.list \
    && printf '%s\n' \
        'Acquire::Retries "5";' \
        'Acquire::http::Timeout "60";' \
        'Acquire::https::Timeout "60";' \
        > /etc/apt/apt.conf.d/99network-timeout

# 编译、调试、SSH和Python工具
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openssh-server ca-certificates curl gnupg \
        build-essential cmake ninja-build pkg-config ccache \
        autoconf automake libtool gdb clangd clang-format \
        git git-lfs rsync \
        unzip zip xz-utils \
        file jq less vim tmux htop tree \
        iproute2 lsof procps \
        python3-dev python3-pip python3-venv python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# 安装Node.js 22
RUN mkdir -p /etc/apt/keyrings \
    && curl -fsSL \
        https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor \
        -o /etc/apt/keyrings/nodesource.gpg \
    && echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && node --version \
    && npm --version

# 安装Python基础工具和uv
RUN python3 -m pip install --no-cache-dir --upgrade \
        pip \
        setuptools \
        wheel \
    && python3 -m pip install --no-cache-dir uv \
    && uv --version

# SSH配置：允许root密钥登录，禁止密码登录
RUN mkdir -p /run/sshd /root/.ssh \
    && chmod 700 /root/.ssh \
    && ssh-keygen -A \
    && sed -ri \
        's/^#?PermitRootLogin.*/PermitRootLogin prohibit-password/' \
        /etc/ssh/sshd_config \
    && sed -ri \
        's/^#?PubkeyAuthentication.*/PubkeyAuthentication yes/' \
        /etc/ssh/sshd_config \
    && sed -ri \
        's/^#?PasswordAuthentication.*/PasswordAuthentication no/' \
        /etc/ssh/sshd_config

RUN mkdir -p \
    /root/WorkSpace \
    /root/.cache/ccache \
    /root/.cache/uv \
    /root/.npm

ENV CCACHE_DIR=/root/.cache/ccache \
    CCACHE_MAXSIZE=10G \
    UV_CACHE_DIR=/root/.cache/uv

WORKDIR /root/WorkSpace

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D", "-e"]
```

接下来就是执行Dockerfile构建新镜像，指定相应的Dockerfile文件名和镜像名即可

```shell
export DOCKER_FILE_NAME=cuda12.4_cudnn.Dockerfile
export DOCKER_IMAGE_NAME=cuda12.4_cudnn_devel
docker build -f ${DOCKER_FILE_NAME} -t ${DOCKER_IMAGE_NAME} .
```

## 3. 启动镜像

一般开发CUDA也会使用到TensorRT，需要根据CUDA环境和版本选择合适的TensorRT，这里直接在[官方](https://developer.nvidia.com/tensorrt/download/10x)下载对应平台的压缩包，并挂载到固定位置。

注意：`--gpus all` 依赖宿主机已安装[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)，未安装的话启动会报 `could not select device driver "" with capabilities: [[gpu]]` 错误。

```bash
export DOCKER_IMAGE_NAME=cuda12.4_cudnn_devel
export DOCKER_HOST_NAME=A10_LINUX
export DOCKER_CONTAINER_NAME=cuda12.4_cudnn_env
docker run -d \
  --name ${DOCKER_CONTAINER_NAME} \
  --hostname ${DOCKER_HOST_NAME} \
  --gpus all \
  --ipc=host \
  --restart unless-stopped \
  -p 1022:22 \
  -v /data/WorkSpace:/root/WorkSpace \
  -v /data/opt/nvidia/TensorRT-10.16.1.11:/opt/nvidia/TensorRT-10.16.1.11:ro \
  ${DOCKER_IMAGE_NAME}
```

## 4. 环境配置

### 1. 将PATH和LD_LIBRARY_PATH添加至系统环境内

```shell
export CUDA_HOME=/usr/local/cuda
export TRT_ROOT=/opt/nvidia/TensorRT-10.16.1.11
export PATH="$CUDA_HOME/bin:$TRT_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$TRT_ROOT/lib:${LD_LIBRARY_PATH:-}"

# 持久化到 ~/.bashrc，重新登录后依然生效
cat >> ~/.bashrc <<'EOF'
export CUDA_HOME=/usr/local/cuda
export TRT_ROOT=/opt/nvidia/TensorRT-10.16.1.11
export PATH="$CUDA_HOME/bin:$TRT_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$TRT_ROOT/lib:${LD_LIBRARY_PATH:-}"
EOF
```

### 2. 配置公钥免密登录

将本机公钥内容复制到docker内的~/.ssh/authorized_keys文件中，即可免密登录。这里我将容器内的22端口映射到了宿主机的1022端口，所以登录时使用

```shell
ssh -p 1022 root@localhost
```
