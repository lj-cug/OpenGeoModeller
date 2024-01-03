# NVHPC Installation Instructions on Ubuntu

https://blog.csdn.net/weixin_45088962/article/details/130346359

Be sure you invoke the install command with the permissions necessary for installing into the desired location.

## using apt install

$ curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg

$ echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | sudo tee /etc/apt/sources.list.d/nvhpc.list

$ sudo apt-get update -y

$ sudo apt-get install -y nvhpc-23-3-cuda-multi

### 指定版本

Bundled with the newest plus two previous CUDA versions (11.4, 11.0, 10.2)

$ curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg

$ echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | sudo tee /etc/apt/sources.list.d/nvhpc.list

$ sudo apt-get update -y

$ sudo apt-get install -y nvhpc-21-9-cuda-multi

## 安装(11.0, 10.2, 10.1)CUDA的nvhpc套件

Bundled with the newest plus two previous CUDA versions (11.0, 10.2, 10.1)

$ wget https://developer.download.nvidia.com/hpc-sdk/20.7/nvhpc-20-7_20.7_amd64.deb \
       https://developer.download.nvidia.com/hpc-sdk/20.7/nvhpc-2020_20.7_amd64.deb \
       https://developer.download.nvidia.com/hpc-sdk/20.7/nvhpc-20-7-cuda-multi_20.7_amd64.deb
	   
$ apt-get install ./nvhpc-20-7_20.7_amd64.deb ./nvhpc-2020_20.7_amd64.deb ./nvhpc-20-7-cuda-multi_20.7_amd64.deb
