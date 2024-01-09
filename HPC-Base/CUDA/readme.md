# CUDA

CUDA��NVIDIA��˾�з���һ��GPGPU���л�����, Ŀǰ��CUDA C/C++/Fortran�ȱ�����

��·���°�����
1. CUDA_Basics   CUDA ��̵�һЩ����֪ʶ
2. GPUDirect     GPU�豸ֱ���������, ������GPUDirect RDMA, GPUDirect Storage��
3. CUDA-aware-MPI   ����֧��GPUDirect��MPI���п�Ľű����ĵ�
4. Ubuntu-GPU     Ubuntu OS��ʹ��CUDA��̵�һЩ����
5. NVHPC-Nvidia   NVIDIA��˾��CUDA�������׼��İ�װ�ű�

## NVHPC-Nvidia

    Nvidia��˾��NVHPC�׼���װ������������OpenACC��CUDA���д�����������

## CUDA-programming

	Nvidia��˾��CUDA��̻���֪ʶ��ʹ��nvcc����������cuԴ���ļ�
	
# CUDA-bashrc
```
# CUDA 11.0 library
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$MY_APPs/cudnn-11.0/lib64:$LD_LIBRARY_PATH

```	