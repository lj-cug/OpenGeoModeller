
## �Կ���صĲ���

### �鿴�Կ��ͺ�����

lspci | grep -i vga

### Ubuntu OS�а�װNVIDIA����

���նˣ�����ָ���Բ鿴���Ե��Կ��ͺţ�

ubuntu-drivers devices

model��Ϊ�Կ����ͺ���Ϣ���˴�ΪGeForce RTX 2070 SUPER���Ƽ����Կ������汾��Ϊnvidia-driver-450 - distro non-free��

### ����ϵͳĬ���Կ�����

gedit /etc/modprobe.d/blacklist.conf

���ļ������ļ�ĩβд�룺

blacklist nouveau

options nouveau modeset=0

������ֶ����£�

sudo update-initramfs -u

### ж��ԭ������

sudo apt-get --purge remove nvidia*

sudo apt autoremove

### �鿴

nvidia-smi