## �ٱ���

��Щ��������ѧϰ�͹������Ѽ���һЩѧϰ���ϣ���Ҫ��Ϊ���о������ܼ��㡢Linux����ϵͳ����̺ͼ���������ѧʹ�á�
Ŀǰ�����ң����ڽ���д��ɵ�ϲ������ĵ���Ȼ��ת�Ƶ� ../ Ŀ¼�������˿���ѧϰ�Ͳ���ʹ�á�

## Ubuntu 20.04��������

### ����

ifconfig -a   //�鿴����������״����eth0�Ƿ���ڣ��ڽ���б�Ӧ���Ҳ���eth0�����ģ�����lo֮�⣬����Ӧ�û���һ��ethX

#### ��̬IP����

vim /etc/network/interfaces //�޸���������
 
auto lo

iface lo inet loopback
 
//������������

auto ethX

iface ethX inet static

address 192.168.1.101

netmask 255.255.255.0

gateway 192.168.1.1

#### ��������

sudo /etc/init.d/networking restart  //������������������������ɽ����

### ��ͬ�汾�ı�����

//��װ�ϰ汾�� gcc9

sudo add-apt-repository ppa:ubuntu-toolchain-r/test

sudo apt-get update

sudo apt-get install gcc-9 g++-9

����gcc

update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 100

update-alternatives --config gcc

���gcc

gcc --version


### �鿴ubuntuӲ����Ϣ

1.�鿴�忨��Ϣ

����cat /proc/pci

2.cpu��Ϣ

	dmidecode -t processor
	
3.Ӳ����Ϣ

�����鿴�������

����fdisk -l

�����鿴��С���

����df -h

�����鿴ʹ�����

����du -h

### FTP����

Putty     WinSCP    FileZilla   Xftp
