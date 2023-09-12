# Cygwin中使用ACE Tools

需要在Cygwin X Server下运行, 
需要安装xinit、xorg_server部件。

启动： 
StartX  或者从开始窗口下运行。

## 安装Cygwin/X需要选择的包

xorg-server (必须, Cygwin/X X 服务器)

xinit (必须, 开启 X server:的脚本 xinit, startx, startwin (包括开始菜单的快捷方式), startxdmcp.bat )

xinit -- -multiplemonitors -multiwindow -clipboard -noprimary -dpi 96 -listen tcp 
( -nolisten  # 不行! )