
## 显卡相关的操作

### 查看显卡型号命令

lspci | grep -i vga

### Ubuntu OS中安装NVIDIA驱动

打开终端，输入指令以查看电脑的显卡型号：

ubuntu-drivers devices

model即为显卡的型号信息，此处为GeForce RTX 2070 SUPER；推荐的显卡驱动版本号为nvidia-driver-450 - distro non-free。

### 禁用系统默认显卡驱动

gedit /etc/modprobe.d/blacklist.conf

打开文件，在文件末尾写入：

blacklist nouveau

options nouveau modeset=0

保存后手动更新；

sudo update-initramfs -u

### 卸载原有驱动

sudo apt-get --purge remove nvidia*

sudo apt autoremove

### 查看

nvidia-smi