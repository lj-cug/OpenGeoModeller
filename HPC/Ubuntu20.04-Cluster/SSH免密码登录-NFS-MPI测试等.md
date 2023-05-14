## SSH�������¼����

���ƽڵ���������  lijian-cug
3������ڵ����������lijian-1   lijian-2   lijian-3

Step 1: ����host�ļ������ƽڵ㣩
$ cat /etc/hosts

127.0.0.1       localhost
192.168.1.86    lijian-cug
192.168.1.150   lijian-1
192.168.1.80    lijian-2
192.168.1.246   lijian-3

Step 2: ����SSH
$ sudo apt-get install openssh-server    # ���ƽڵ��ϰ�װssh������

##����ssh��¼
ssh username@hostname   # ��ʱ��¼����Ҫ����username�����룬

##Ϊ���������¼ssh����Ҫ����keys��Ȼ���Ƶ�����ڵ������authorized_keys�б���
$ ssh-keygen -t rsa
$ ssh-copy-id 192.168.1.150      # ��һ��copy����Ҫ�������� 
$ ssh-copy-id 192.168.1.80
$ ssh-copy-id 192.168.1.246

##��������ssh��¼
$ eval 'ssh-agent'
$ ssh-add ~/.ssh/id_dsa

##ssh�������¼����
$ ssh lijian-1    # lijian-2  lijain-3

Step 4: ���� NFS
# NFS-server
$ sudo apt-get install nfs-kernel-server   # ���ƽڵ�

$ mkdir nfs   # �������繲���ļ���

# ���nfs�����ļ���
$  cat /etc/exports
/home/lijian/nfs 192.168.1.150(rw,sync,no_root_squash,no_subtree_check)
/home/lijian/nfs 192.168.1.225(rw,sync,no_root_squash,no_subtree_check)

$ exportfs -a

# ����NFS���ƽڵ�
$ sudo service nfs-kernel-server restart

# NFS����ڵ�
$ sudo apt-get install nfs-common
$ mkdir /home/lijian/nfs -p

# ���ع����ļ���
$ sudo mount -t nfs 192.168.1.86:/home/lijian/nfs /home/lijian/nfs

# �����ص�·��
$ df -h
Filesystem      		    Size  Used Avail Use% Mounted on
manager:/home/lijian/nfs  49G   15G   32G  32% /home/lijian/nfs

# ʹ����ڵ��ϵ�nfs����������Ч
$ cat /etc/fstab
192.168.1.86:/home/lijian/nfs /home/lijian/nfs nfs defaults,_netdev 0 0 

Step 5: ��������MPI��ִ�г���
$ mpicc -o mpi_sample mpi_sample.c    # ���������mpi����

# ����ִ�г��򿽱���lijian-cug�Ĺ����ļ���nfs��
$ cd nfs/
$ pwd
/home/lijian/nfs

# �ڿ��ƽڵ�������
$ mpirun -np 2 ./cpi     # No. of processes = 2

# �ڼ�Ⱥ������
$ mpirun -np 5 -hosts worker,localhost ./cpi
#hostnames can also be substituted with ip addresses.

# ���ߣ���hostfile�ж���ip��ַ��hostname
$ mpirun -np 5 --hostfile mpi_file ./cpi

�����Ĵ���ͽ���
(1) ���л����϶���װ��ͬ��MPIͨ�ſ�
(2) manager��hosts�ļ�Ӧ������������IP��ַ������worker�ڵ��IP��ַ�����磺
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

��worker�ڵ㣬��Ҫmanager��ڵ�IP��ַ�Ͷ�Ӧ��worker�ڵ��IP��ַ�����磺
$ cat /etc/hosts
127.0.0.1	localhost
#127.0.1.1	1947

#MPI CLUSTER SETUP
172.50.88.22	manager
172.50.88.54	worker3

(3)MPI���п�ִ�г��򣬿��Ի�����е��غ�Զ�̽ڵ㣬�����ܽ�����Զ�̽ڵ���㡣
# ���������OK��
$ mpirun -np 10 --hosts manager ./cpi
# To run the program only on the same manager node

# ���������OK��
$ mpirun -np 10 --hosts manager,worker1,worker2 ./cpi
# To run the program on manager and worker nodes.

# ��������в����У�
$ mpirun -np 10 --hosts worker1 ./cpi
# Trying to run the program only on remote worker

(4) hostfile�ı�д�ɲο���https://blog.51cto.com/u_15642578/5316847
��򵥵�hostfile
# slots���Բ�д��Ĭ��ʹ��ÿ������ڵ��CPU��ʵ��������ģ�
# n1,n2,n3,n4������hostname��Ҳ������IP��ַ
n1 slots=1
n2 slots=2
n3 slots=2
n4 slots=4

(5) --hostfile ?   --machinefile ?