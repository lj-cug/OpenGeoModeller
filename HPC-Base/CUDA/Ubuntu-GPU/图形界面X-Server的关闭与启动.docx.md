# 图形界面 X Server 的关闭与启动

Linux图形界面多数使用的是 X Server, 我们有时需要关闭/重启它. 比如: 安装
NVIDIA 的驱动程序时，就需要先关闭 X server; 希望让系统以 server
方式运行,关闭桌面环境以降低不必要的性能损耗.

## Ubuntu20.04

ubuntu20 的默认桌面环境管理服务是 gdm3, 停止 gdm3 就关闭了图形界面(GUI).

在不关机情况下关闭和打开图形界面

\# 在的当前会话中关闭图形界面 (如果看不到命令行界面，按 ctrl + alt +
\[1-7\] 试试)

sudo systemctl stop gdm3

\# 重新打开图形界面

sudo systemctl start gdm3

操作发现, ubutu20 图形界面默认在 tty1 上启动, tty2-6 默认是命令行,
tty7打不开(不晓得哪里姿势不对)

设置开机默认进入GUI或命令行

\# 设置开机默认进入命令行

sudo systemctl set-default multi-user.target

sudo reboot \# 重启看看

\# 设置开机默认进入用户图形界面

sudo systemctl set-default graphical.target

sudo reboot \# 重启看看

## Ubuntu18.04

Ubuntu18.04 虽然默认使用了gnome桌面，但是经过测试 gdm
并不能很好得工作，通过设置系统启动方式，然后重启来达到关闭 x server
的目的

ps. gdm是GNOME Display Manager, GNOME的桌面环境管理器

\# 关闭用户图形界面

sudo systemctl set-default multi-user.target

sudo reboot

\# 开启用户图形界面

sudo systemctl set-default graphical.target

sudo reboot

## Ubuntu16.04 管理 x server

1.  用gdm管理

sudo /etc/init.d/gdm stop

sudo /etc/init.d/gdm status

sudo /etc/init.d/gdm restart

如果 /etc/init.d 下面没有 gdm 的话，可以尝试另一种方法

2.  用lightdm管理

sudo /etc/init.d/lightdm stop

sudo /etc/init.d/lightdm status

sudo /etc/init.d/lightdm restart

3.  用service管理

sudo service lightdm stop

sudo service lightdm status

sudo service lightdm start

## centos7 管理 x server

sudo systemctl stop gdm.service

sudo systemctl status gdm.service

sudo systemctl start gdm.service
