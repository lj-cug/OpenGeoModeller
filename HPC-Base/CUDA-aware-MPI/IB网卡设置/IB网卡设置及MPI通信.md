##Mellanox��������

���PCIE �Ƿ�ʶ��IB��
lspci | grep -i Mellanox*

���IB״̬
ibstat      // ib��StateΪ active ���� Link Layer Ϊ: InfiniBand �����������Link Layer Ϊ Ethernet ģʽ

������ʱIP
sudo ifconfig ib0 11.1.1.15 up     //ib0Ϊ��һ��IB���� ip��ַ�Լ�����

���Զ����� - 
ǰ��: Server�˺� Client ����ͨ����ib��������ip��ַ
 - mlx5 �����iib�����ͺţ������ͺŸ�������� ibstat ����ʾ��Ϊ׼ 
 - ib_read_bw ��ib����װ���Դ������� 
 - ���������������£� 
	- ib_atomic_bw ib_atomic_lat ib_read_bw ib_read_lat ib_send_bw ib_send_lat ib_write_bw ib_write_lat

1.	ib_read_bw  //server ��ִ��
2.	ib_read_bw -d mlx5_0 -a -F -i 1 11.1.1.15  // ip��ַ������������ip��ַ,Ҳ������IB���� IP��ַ
3.	�ȴ�������


�л�IB��ģʽΪ InfiniBand
1. sudo mst start 
2. sudo mlxconfig  -y -d /dev/mst/mt4119_pciconf0 set LINK_TYPE_P1=1 
3. sudo reboot
3. ibstat  // �鿴�޸ĺ��IB��ģʽ


�鿴IB ��Ӳ���ͺ���Ϣ
mlxvpd -d mlx5_0  // -d  Ϊ ib  hca_id, ����ͨ��ibstat�в鿴

NUMA �ܹ���IB�������ȶ��������
// server ��ִ��
$ cat /sys/class/net/ib0/device/numa_node 
$ 3
$ numactl --cpunodebind=3 --membind=3  ib_read_bw -d mlx5_0

// client ��ִ��
$ cat /sys/class/net/ib0/device/numa_node 
$ 7
$ numactl --cpunodebind=7 --membind=7 ib_read_bw -a -F -d mlx5_0 10.0.0.1 --run_infinitely



##MPI��ʹ��IB��ͨ��

-np N   ����N������
-hostfile   ָ������ڵ㣬��ʽ���£�
node1 slots=8
node2 slots=8

##ѡ��ͨ������
OpenMPI֧�ֶ���ͨ��Э�飬����̫����IB���硢�����ڴ�ȡ���ͨ������--mca btl��������ѡ���磺
mpirun -np 12 --ostfile hosts --mca btl self,sm,openib ./pro.exe

--mca btl self,tcp  # ʹ��TCP/IPЭ��ͨ��
--mca btl����tcp_if_include etho   #��̫��ͨ��ʱ��ʹ��eth0�ӿڣ�Ĭ��ʹ�����нӿ�
--mca oret_rsh-agent rsh    #ָ���ڵ��ͨ��ʹ��rsh��Ĭ��Ϊssh




