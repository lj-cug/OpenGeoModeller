# 高性能计算(HPC)基座

## Linux-Linux-garbage

	Linux系统命令百宝箱，为了自己研究时方便查阅
	
	[CFD-CUG本科教学使用的常用工具](./Linux-garbage/Tools)
	
## Ubuntu20.04-Cluster	

	我自己搭建的集群的硬件和软件配置和操作。
	
	按照其中的文档说明，你也可以快速地搭建一个属于你自己的“超级计算机"
			
## Programming&Compilation	

	编程与编译方法
		
## Nvidia-GPGPU
	
	基于Nvidia GPU的通用异构并行环境搭建手册

## GPUDirect

    Nvidia公司的GPU设备直连技术
	
## CUDA-aware MPI

	支持GPUDirect技术的MPI并行通信库的编译，就是OpenMPI和MVMPICH2
	
## PETSc
	
	HPC应用中常用的PETSc库的编译说明
	
## Solver

    线性方程组(Ax=b)求解器
	
### Amgx
	
	Nvidia公司开源的GPU集群并行求解器，与PETSc库结合使用：PETSc-Amgx
	
### AmgCL

	俄国人开发的多重网格求解器，使用OpenCL实现异构并行
	
### paralution

	只能用于单节点计算，集群版本是收费的商业版本
	
## HPX

    异步并行运行时系统
	
## Domain-Decomposition

	非结构网格的区域分解库，常用的有：METIS, SCOTCH, Zoltan等
   
## HPC-IO

	IO是HPC环境下的一个瓶颈，MPI-IO，GDS等技术可用于解决此类问题

## Numerical_Algorithm

	HPC应用的数值算法
