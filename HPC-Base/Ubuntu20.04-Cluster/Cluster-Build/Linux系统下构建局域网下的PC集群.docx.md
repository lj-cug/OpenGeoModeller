# Linux系统下构建局域网下的PC集群

## 4 两台PC的"集群"

## 4.1 硬件

两台节点：intel i7
4核，6g内存，1T硬盘，独立显卡（gpu太弱，不用它来计算，只用于连接显示器）intel
Gigabit以太网卡。

网络连接：gigabit以太网switch hub（L2交换机），网线，需要连接互联网。

## 4.2 安装操作系统

安装ubuntu，具体的安装步骤就很简单，略。但是要注意的是尽量采用同样的用户名和密码，并将计算机名称编号，最后记得把用户设成管理员。

## 4.3 配置环境，安装软件

sudo apt-get -y update

sudo apt-get -y upgrade

首先，更新一下apt。

sudo apt -y install openssh-server openssh-client nfs-common libgomp1
openmpi-bin libopenmpi-dev openmpi-common update-manager-core

然后安装这些软件...

openssh-server openssh-client: openssh 的服务和客户端

nfs-common: 网络文件系统

openmpi-bin libopenmpi-dev openmpi-common: openmpi，开源的mpi

update-manager-core: 更新用

sudo apt-get -y dist-upgrade

然后，更新一下软件的依赖，防止出现依赖的遗漏。

## 4.4 网络配置

a）网络连接，使用交换机集线器连接，连接方式多种多样，由于我们的组装的规模很小，所以，低于交换机的接口数的时候就把交换机作为行星网络的中心，如果需要多个交换机，就采用层状交换机结构。出于简单考虑，IP没有设成静态，直接采用路由器的hdcp功能，动态分配ip，如果要长期使用，请设置成静态ip。那么自动分配的ip分别是[XXX.65.121.82和
XXX.65.121.102]{.mark}，分别对应的名称是[server01和server02]。[更改静态ip需要更改]
／etc／network／interface
文件（Ubuntu18的这部分设置被更新了，需要采取其他方法）。

b）查询ip地址，可以直接去gui的界面去看，也可以输入\<ifconfig\>命令。

c）更改／etc／hosts文件，把ip和名字对应上，这样操作起来比较方便，不用处处都输入ip。如果查看自己的hostname，可以查看／etc／hostname文件。

d）然后重启network服务

sudo /etc/init.d/networking restart

测试两台机器是否能够ping通，如果，可以，说明网络没问题，如果出现问题，检查网络连接和网络设置。

e）用ssh生成密钥

ssh-keygen

f）同步密钥，可以进行免密ssh登陆。

cd \~/.ssh

cp id_rsa.pub authorized_keys

rsync -avz \~/.ssh XXX.65.121.102:\~/.

再测试两台机器是否可以免密ssh登陆，如果不行，检查问题。

这时候有人会问，我现在有两台机器，可以拷贝一下密钥，要是我有n台机器，岂不是相互都要拷贝n-1个密钥？解决的方法很简单就是大家都用一个密钥，这样进行访问的时候就不用互相交叉进行，而是由一个中心进行相互通信。当然这就失去了非对称加密的意义，理论上利用rsh的话会更好一些，加密毕竟会增加一些通信时间。

\*因为mpi通信需要网络权限，最好关闭防火墙和网络管理器。

## 4.6多机并行

新建一个mpitest.c 文件

拷贝编译完成的a.out，到另外一台机器，保证两台机器的a.out出现在同一个目录位置。

scp \~/a.out XXX.65.121.102:\~/a.out

然后编辑machinefile

server018 cpu=4
server156 cpu=4

然后在这个目录下运行a.out（确保a.out也在这个文件夹下）

%mpirun \--machinefile machinefile -np 8 a.out

Hello world from processor server018, rank 3 out of 8 processors
Hello world from processor server018, rank 0 out of 8 processors
Hello world from processor server018, rank 1 out of 8 processors
Hello world from processor server018, rank 2 out of 8 processors
Hello world from processor server156, rank 6 out of 8 processors
Hello world from processor server156, rank 5 out of 8 processors
Hello world from processor server156, rank 4 out of 8 processors
Hello world from processor server156, rank 7 out of 8 processors

这样就是十分初级的集群的雏形，可以进行多机并列计算。

## 4.7 NFS设置

刚我们进行计算中的重要一步就是把编译好的可执行程序copy到每一台机器上，并保证其目录的位置相同，为了避免这样的重复操作，我们[最好使用共享的文件系统]{.mark}，最简单的方式就是设置NSF。依靠网络进行挂载可以保证每一台节点都可以访问相同的路径。

(1)先安装相关软件

yum install nfs-utils nfs-utils-lib

(2)设置nfs相关服务在操作系统启动时启动

systemctl enable rpcbind

systemctl enable nfs-server

systemctl enable nfs-lock

systemctl enable nfs-idmap

(3)启动nfs服务

systemctl start rpcbind

systemctl start nfs-server

systemctl start nfs-lock

systemctl start nfs-idmap

(4)服务器端设置NFS卷输出，即编辑 /etc/exports 添加：

sudo emacs /etc/exports

/nfs XXX.65.121.0/24(rw)

/nfs -- 共享目录

\<ip\>/24 -- 允许访问NFS的客户端IP地址段

rw -- 允许对共享目录进行读写

当然还有其他设置选项，比如insecure sync \...

sudo exportfs

这个是显示一下是否挂在成功

service nfs status -l

(5)查看NFS状态

service nfs restart

重启NFS，这样，服务器端就设定结束了。

Linux挂载NFS的客户端非常简单的命令，先创建挂载目录，然后用 -t nfs
参数挂载就可以了:

mount -t nfs xxx.168.0.100:/nfs /nfs

可以先查看

showmount -e 192.168.0.100

如果要设置客户端启动时候就挂载NFS，可以配置 /etc/fstab 添加以下内容

sudo emacs /etc/fstab

192.168.0.100:/nfs /nfs defaults 0 0

然后在客户端简单使用以下命令就可以挂载

mount /nfs
