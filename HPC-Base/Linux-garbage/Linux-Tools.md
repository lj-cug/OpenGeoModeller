# Linux-Tools

  Linuxϵͳ�Ƽ�ʹ�� Ubuntu 20.04
  ��Windowxϵͳ�ıʼǱ��ϣ�����ʹ��VMware workstation (��ҵ��������ƽ�棩������ʹ��[VirtualBox](https://www.virtualbox.org/)
  
# Ubuntu 20.04ʹ�÷���
  
  [���ز���װUbuntu 22.04](http://releases.ubuntu.com/22.04/ubuntu-22.04.3-desktop-amd64.iso)
  
## ��һ��ʹ��su (�����û�Ȩ��)
   `sudo passwd root`
   `�����¾�����`
   `Ȼ��, su`
   
## �޸�Ϊ����Դ
	1.����������鿴�汾����
      lsb_release -c
	  
	2.����ԭ����Դ������ǰ��Դ����һ�£��Է��Ժ���Ҫ�õ�
      sudo cp /etc/apt/sources.list /etc/apt/sources.list1
   
	3.��/etc/apt/sources.list�ļ������Դ��������
      sudo gedit /etc/apt/sources.list
	  
	4.�����������ݵ��ļ�sources.list��
`deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse`
`deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse`
`deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse`
`deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-security main restricted universe multiverse`

	5.ճ����֮��ֱ���˳�����

	6.ִ���������
	sudo apt-get update
	sudo apt-get upgrade

## ��װGNU������

	apt-get install build-essential gfortran       
   
## ��װopenMPI���п�
   
    apt-get install mpi-default-dev openmpi-bin openmpi-common
  
## ��װ��ƽ̨���빤��

    apt-get install make cmake cmake-gui
   
# bash�Ļ���ʹ�÷���
    
	�ο�"Bash����Ļ�������.md"
   
# Conda���⻷��  

   ����ʹ��Miniconda����Python3�����⻷������"Miniconda����ʹ��.md"

# Tecplot���ӻ����
  
  Tecplot360.2009�ǱȽ��ϵ�Tecplot�������ҵ���ƽ�棩�������װ�װ��
  ��2018�汾��Tecplot360�����ƽ���̱Ƚϸ���һ�㡣

  