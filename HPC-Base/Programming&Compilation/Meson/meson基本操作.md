# Meson基本操作

Meson旨在开发最具可用性和快速的构建系统。提供简单但强大的声明式语言用来描述构建。原生支持最新的工具和框架，如Qt5、代码覆盖率、单元测试和预编译头文件等。利用一组优化技术来快速编译代码，包括增量编译和完全编译。

## ubuntu上安装meson
   
    安装python3和ninja:

	apt-get install python3 python3-pip ninja-build

	pip install meson ninja
	
	在meson.build目录执行： 	
	meson build
	
	进入 build目录，执行ninja
	
	cd ninja && ninja
	
## meson-ui (MESON可视化界面)

Qt GUI for the Meson build system

https://github.com/michaelbrockus/meson-ui

pip install meson-ui

## meson.build编写

编译可执行程序：

project('project01', 'c')
executable("project", 'src/main.c')

编译静态链接库：

project('project02', 'c')
static_library('thirdinfo', 'src/third_lib.c')


