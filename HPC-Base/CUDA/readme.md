# CUDA

CUDA是NVIDIA公司研发的一种GPGPU并行化语言, 目前有CUDA C/C++/Fortran等编译器

本路径下包含：
1. CUDA_Basics   CUDA 编程的一些基础知识
2. GPUDirect     GPU设备直连技术编程, 包含：GPUDirect RDMA, GPUDirect Storage等
3. CUDA-aware-MPI   编译支持GPUDirect的MPI并行库的脚本和文档
4. Ubuntu-GPU     Ubuntu OS中使用CUDA编程的一些技巧
5. NVHPC-Nvidia   NVIDIA公司的CUDA编译器套件的安装脚本

## NVHPC-Nvidia

    Nvidia公司的NVHPC套件安装及开发，用于OpenACC等CUDA并行代码编译和运行

## CUDA-programming

	Nvidia公司的CUDA编程基础知识，使用nvcc编译器编译cu源码文件
	
# CUDA-bashrc
```
# CUDA 11.0 library
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$MY_APPs/cudnn-11.0/lib64:$LD_LIBRARY_PATH

```	