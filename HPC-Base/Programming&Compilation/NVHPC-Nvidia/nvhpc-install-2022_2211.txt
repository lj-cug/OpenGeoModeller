# Install-NVHPC-22.11

wget https://developer.download.nvidia.com/hpc-sdk/22.11/nvhpc_2022_2211_Linux_x86_64_cuda_multi.tar.gz

tar xpzf nvhpc_2022_2211_Linux_x86_64_cuda_multi.tar.gz

cd nvhpc_2022_2211_Linux_x86_64_cuda_multi

sudo ./install

## 设置环境变量

export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/compilers/bin/:$PATH

export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/compilers/lib/:$LD_LIBRARY_PATH

export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/comm_libs/mpi/bin/:$PATH

export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/comm_libs/mpi/lib/:$LD_LIBRARY_PATH
