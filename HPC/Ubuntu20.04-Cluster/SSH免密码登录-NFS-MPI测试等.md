## SSH免密码登录设置

控制节点主机名：  lijian-cug
3个计算节点的主机名：lijian-1   lijian-2   lijian-3

Step 1: 配置host文件（控制节点）
$ cat /etc/hosts

127.0.0.1       localhost
192.168.1.86    lijian-cug
192.168.1.150   lijian-1
192.168.1.80    lijian-2
192.168.1.246   lijian-3

Step 2: 设置SSH
$ sudo apt-get install openssh-server    # 控制节点上安装ssh服务器

##测试ssh登录
ssh username@hostname   # 此时登录，需要输入username的密码，

##为了免密码登录ssh，需要生成keys，然后复制到计算节点机器的authorized_keys列表中
$ ssh-keygen -t rsa
$ ssh-copy-id 192.168.1.150      # 第一次copy，需要输入密码 
$ ssh-copy-id 192.168.1.80
$ ssh-copy-id 192.168.1.246

##启动密码ssh登录
$ eval 'ssh-agent'
$ ssh-add ~/.ssh/id_dsa

##ssh免密码登录测试
$ ssh lijian-1    # lijian-2  lijain-3

Step 4: 设置 NFS
# NFS-server
$ sudo apt-get install nfs-kernel-server   # 控制节点

$ mkdir nfs   # 创建网络共享文件夹

# 输出nfs共享文件夹
$  cat /etc/exports
/home/lijian/nfs 192.168.1.150(rw,sync,no_root_squash,no_subtree_check)
/home/lijian/nfs 192.168.1.225(rw,sync,no_root_squash,no_subtree_check)

$ exportfs -a

# 重启NFS控制节点
$ sudo service nfs-kernel-server restart

# NFS计算节点
$ sudo apt-get install nfs-common
$ mkdir /home/lijian/nfs -p

# 挂载共享文件夹
$ sudo mount -t nfs 192.168.1.86:/home/lijian/nfs /home/lijian/nfs

# 检查挂载的路径
$ df -h
Filesystem      		    Size  Used Avail Use% Mounted on
manager:/home/lijian/nfs  49G   15G   32G  32% /home/lijian/nfs

# 使计算节点上的nfs挂载永久生效
$ cat /etc/fstab
192.168.1.86:/home/lijian/nfs /home/lijian/nfs nfs defaults,_netdev 0 0 

Step 5: 测试运行MPI可执行程序
$ mpicc -o mpi_sample mpi_sample.c    # 编译测试用mpi程序

# 将可执行程序拷贝到lijian-cug的共享文件夹nfs下
$ cd nfs/
$ pwd
/home/lijian/nfs

# 在控制节点上运行
$ mpirun -np 2 ./cpi     # No. of processes = 2

# 在集群上运行
$ mpirun -np 5 -hosts worker,localhost ./cpi
#hostnames can also be substituted with ip addresses.

# 或者，在hostfile中定义ip地址或hostname
$ mpirun -np 5 --hostfile mpi_file ./cpi

常见的错误和建议
(1) 所有机器上都安装相同的MPI通信库
(2) manager的hosts文件应包含本地网络IP地址和所有worker节点的IP地址，例如：
$ cat /etc/hosts
127.0.0.1	localhost
#127.0.1.1	1944

#MPI CLUSTER SETUP
172.50.88.22	manager
172.50.88.56 	worker1
172.50.88.34 	worker2
172.50.88.54	worker3
172.50.88.60 	worker4
172.50.88.46	worker5

各worker节点，需要manager入口的IP地址和对应的worker节点的IP地址，例如：
$ cat /etc/hosts
127.0.0.1	localhost
#127.0.1.1	1947

#MPI CLUSTER SETUP
172.50.88.22	manager
172.50.88.54	worker3

(3)MPI运行可执行程序，可以混合运行当地和远程节点，但不能仅运行远程节点计算。
# 下面的运行OK！
$ mpirun -np 10 --hosts manager ./cpi
# To run the program only on the same manager node

# 下面的运行OK！
$ mpirun -np 10 --hosts manager,worker1,worker2 ./cpi
# To run the program on manager and worker nodes.

# 下面的运行不可行！
$ mpirun -np 10 --hosts worker1 ./cpi
# Trying to run the program only on remote worker

(4) hostfile的编写可参考：https://blog.51cto.com/u_15642578/5316847
最简单的hostfile
# slots可以不写（默认使用每个计算节点的CPU的实际物理核心）
# n1,n2,n3,n4可以是hostname，也可以是IP地址
n1 slots=1
n2 slots=2
n3 slots=2
n4 slots=4

(5) --hostfile ?   --machinefile ?