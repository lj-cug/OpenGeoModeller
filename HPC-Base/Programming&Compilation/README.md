# Compilation 

	源代码编译和编译器的总结
	
	不同的处理器厂商，有不同的编译器，可针对处理器的架构实施优化，提高程序的执行效率，例如：
	
	Huawei公司的Bisheng编译器
	
	AMD公司的AOCC编译器
	
	Intel公司的Intel OneAPI编译器
	
	Nvidia公司的HPC SDK编译器(收购了PGI编译器)
	
	...  ...
	
	总之，大部分自研编译器都是在clang和flang编译器基础做的优化
	
## OneAPI-Intel

    Intel公司的OneAPI套件安装及开发，即Intel HPC套件
	
## NVHPC-Nvidia

    Nvidia公司的NVHPC套件安装及开发，用于OpenACC等CUDA并行代码编译和运行

## CUDA-programming

	Nvidia公司的CUDA编程基础知识，使用nvcc编译器编译cu源码文件
	
## OpenCL

	Compute Language, 支持CPU和GPU的多线程并行
	
## 编译系统

	可使用GNU编译器，如: gcc, g++, gfortran；也可使用一些厂商研发的编译器
	
	不同的源代码，使用makefile, CMAKE, SCON和MESON等编译工具

### Makefile

	最基本的编译系统；Makefile语法可参考"陈皓. 跟我一起写 Makefile"

### CMAKE
	
	跨平台编译系统；有cmake-gui
	
### SCons

	基于Python的编译系统
	
### Meson
	
	基于Python和ninja的编译系统；有GUI界面的meson-gui
	
## Docker容器

	快速部署复杂程序的编译和运行环境的容器
	
## Singularity容器

	面向HPC环境的容器
	
## Compilers

	Windows 与Linux OS下的编译器以及编译参数总结
	
## Programming-Language

	编译语言相关的文档


	