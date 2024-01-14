# Linux-Tools

  Linux系统推荐使用 Ubuntu 20.04
  在Windowx系统的笔记本上，可以使用VMware workstation (商业软件，有破解版），或者使用[VirtualBox](https://www.virtualbox.org/)
  
# Ubuntu 20.04使用方法
  
  [下载并安装Ubuntu 22.04](http://releases.ubuntu.com/22.04/ubuntu-22.04.3-desktop-amd64.iso)
  
## 第一次使用su (超级用户权限)
   `sudo passwd root`
   `输入新旧密码`
   `然后, su`
   
## 修改为国内源
	1.用以下命令查看版本名：
      lsb_release -c
	  
	2.备份原来的源，将以前的源备份一下，以防以后需要用的
      sudo cp /etc/apt/sources.list /etc/apt/sources.list1
   
	3.打开/etc/apt/sources.list文件，添加源，并保存
      sudo gedit /etc/apt/sources.list
	  
	4.复制下面内容到文件sources.list中
`deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse`
`deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse`
`deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse`
`deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-security main restricted universe multiverse`

	5.粘贴完之后，直接退出保存

	6.执行命令更新
	sudo apt-get update
	sudo apt-get upgrade

## 安装GNU编译器

	apt-get install build-essential gfortran       
   
## 安装openMPI并行库
   
    apt-get install mpi-default-dev openmpi-bin openmpi-common
  
## 安装跨平台编译工具

    apt-get install make cmake cmake-gui
   
# bash的基本使用方法
    
	参考"Bash命令的基本方法.md"
   
# Conda虚拟环境  

   建议使用Miniconda构建Python3的虚拟环境，见"Miniconda基本使用.md"

# Tecplot可视化软件
  
  Tecplot360.2009是比较老的Tecplot软件（商业，破解版），但容易安装。
  有2018版本的Tecplot360，但破解过程比较复杂一点。

  