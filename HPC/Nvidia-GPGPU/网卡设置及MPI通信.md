##Mellanox网卡设置

检查PCIE 是否识别IB卡
lspci | grep -i Mellanox*

检查IB状态
ibstat      // ib卡State为 active 并且 Link Layer 为: InfiniBand 则正常，如果Link Layer 为 Ethernet 模式

配置临时IP
sudo ifconfig ib0 11.1.1.15 up     //ib0为第一块IB卡， ip地址自己定义

测试读带宽 - 
前提: Server端和 Client 网络通畅，ib卡已配置ip地址
 - mlx5 代表的iib卡的型号，具体型号根据请根据 ibstat 中显示的为准 
 - ib_read_bw 是ib卡安装包自带的命令 
 - 其他测试命令如下： 
	- ib_atomic_bw ib_atomic_lat ib_read_bw ib_read_lat ib_send_bw ib_send_lat ib_write_bw ib_write_lat

1.	ib_read_bw  //server 端执行
2.	ib_read_bw -d mlx5_0 -a -F -i 1 11.1.1.15  // ip地址可以是网卡的ip地址,也可以是IB卡的 IP地址
3.	等待结果输出


切换IB卡模式为 InfiniBand
1. sudo mst start 
2. sudo mlxconfig  -y -d /dev/mst/mt4119_pciconf0 set LINK_TYPE_P1=1 
3. sudo reboot
3. ibstat  // 查看修改后的IB卡模式


查看IB 卡硬件型号信息
mlxvpd -d mlx5_0  // -d  为 ib  hca_id, 可以通过ibstat中查看

NUMA 架构下IB卡带宽不稳定解决方法
// server 端执行
$ cat /sys/class/net/ib0/device/numa_node 
$ 3
$ numactl --cpunodebind=3 --membind=3  ib_read_bw -d mlx5_0

// client 端执行
$ cat /sys/class/net/ib0/device/numa_node 
$ 7
$ numactl --cpunodebind=7 --membind=7 ib_read_bw -a -F -d mlx5_0 10.0.0.1 --run_infinitely



##MPI库使用IB卡通信

-np N   运行N个进程
-hostfile   指定计算节点，格式如下：
node1 slots=8
node2 slots=8

##选择通信网络
OpenMPI支持多种通信协议，如以太网、IB网络、共享内存等。可通过设置--mca btl参数进行选择，如：
mpirun -np 12 --ostfile hosts --mca btl self,sm,openib ./pro.exe

--mca btl self,tcp  # 使用TCP/IP协议通信
--mca btl——tcp_if_include etho   #以太网通信时，使用eth0接口，默认使用所有接口
--mca oret_rsh-agent rsh    #指定节点间通信使用rsh，默认为ssh




