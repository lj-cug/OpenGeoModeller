# singularity的安装和使用

## 安装依赖

sudo apt-get update && sudo apt-get install -y 
build-essential 
uuid-dev  
libgpgme-dev 
squashfs-tools  
libseccomp-dev  
cryptsetup-bin

## 安装GO语言

下载地址：https://golang.org/dl/

wget https://studygolang.com/dl/golang/go1.21.1.src.tar.gz     #下载

tar -C /pub/software -xzvf go1.21.1.src.tar.gz                 #解压

rm go1.21.1.src.tar.gz                                #删除安装包

### 添加到环境变量

echo 'export PATH=/pub/software/go/bin(你的路径):$PATH' >> ~/.bashrc

## 下载singularity

下载地址：https://github.com/hpcng/singularity/releases

wget https://github.com/hpcng/singularity/releases/download/v3.7.2/singularity-3.7.2.tar.gz

tar -xzf singularity-3.7.2.tar.gz #解压

cd singularity

## 安装singularity

./mconfig

cd builddir

make

sudo make install

记得添加到环境变量

# 快速上手

## 下载镜像

可以从 Container Library（https://cloud.sylabs.io/library）

例如：

singularity pull library://cenat/default/blast.sif:latest

or Docker Hub（https://hub.docker.com/)下载images。

例如：

singularity pull docker://ncbi/blast

singularity pull --arch amd64 library://library/default/ubuntu:20.04

### 进入容器

默认会自动挂载home/PWD , /tmp , /proc , /sys , /dev 目录。

singularity shell --writable --fakeroot blast

在容器中安装软件，建议不要使用anaconda 安装，而是手动安装，我们要尽量保持容器轻量。

添加环境变量

退出容器后, 在blast/environment 中添加PATH

"
vi blast/environment
!/bin/sh
export PATH=/opt/ncbi-blast-2.10.1+/bin:$PATH
"

### 打包

软件全部安装完成之后将容器打包

singularity build blast.sif blast

### 运行程序

singularity exec blast.sif  blasp XXX 后面接软件的用法

## 运行容器

### 交互式运行

singularity shell blast.sif bash

### 直接运行

singularity exec blast.sif blastp

### 用户和权限

使用容器不得不考虑安全性，安全性来自两个方面，一个是使用了不被信任的容器，这个就像你在电脑上安装了不被信任的软件一样，Singularity提供了签名机制来验证；
另一方面是容器会不会越权对Host做一些不该做的事情，这个是需要考虑的。

singularity 的解决办法是会在容器内动态创建一个用户，该用户与Host里的用户名、组名、权限等都保持一致。
这样你在Host 中做不了的事情，在容器里也干不了。