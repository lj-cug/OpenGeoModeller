## 百宝箱

这些年来，我学习和工作中搜集的一些学习资料，主要是为了研究高性能计算、Linux操作系统、编程和计算流体力学使用。
目前很杂乱，后期将编写“傻瓜操作”文档，然后转移到 ../ 目录，共他人快速学习和操作使用。

## Ubuntu 20.04基本操作

### 网络

ifconfig -a   //查看所有网卡现状，看eth0是否存在，在结果列表应该找不到eth0网卡的，除了lo之外，正常应该还有一个ethX

#### 静态IP设置

vim /etc/network/interfaces //修改内容如下
 
auto lo

iface lo inet loopback
 
//增加如下内容

auto ethX

iface ethX inet static

address 192.168.1.101

netmask 255.255.255.0

gateway 192.168.1.1

#### 重启网络

sudo /etc/init.d/networking restart  //重新启动网卡，理论上问题可解决。

### 不同版本的编译器

//安装老版本的 gcc9

sudo add-apt-repository ppa:ubuntu-toolchain-r/test

sudo apt-get update

sudo apt-get install gcc-9 g++-9

设置gcc

update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 100

update-alternatives --config gcc

检查gcc

gcc --version


### 查看ubuntu硬件信息

1.查看板卡信息

　　cat /proc/pci

2.cpu信息

	dmidecode -t processor
	
3.硬盘信息

　　查看分区情况

　　fdisk -l

　　查看大小情况

　　df -h

　　查看使用情况

　　du -h

### FTP工具

Putty     WinSCP    FileZilla   Xftp
