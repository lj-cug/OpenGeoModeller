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

### 重启网络
sudo /etc/init.d/networking restart  //重新启动网卡，理论上问题可解决。