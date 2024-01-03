# ACE-tools

主要是使用xmgredit5生成带边界信息的SCHISM网格输入文件.

[ACE-tools主页](http://ccrm.vims.edu/w/index.php/ACE_tools)

ACE tools (xmgr5, xmgredit5, and xmvis6) are custom-made for SCHISM model, and we highly recommend you install them on your local system. Compared to other visualization tools, ACE tools are highly efficient and are sufficient for your day-to-day research. However, note that some new features of SCHISM are not fully supported by ACE, and so alternative packages (VisIT) should be used.

The tools can be installed on linux or windows systems.

xmgredit5: a light-weight grid generation and full fledged grid editor. Very useful for creating input files for SCHISM;  (用于对grd网格文件, 生成边界标记等, 生成供SCHISM使用的输入文件)

xmvis6: viz tool for the binary outputs from SCHISM (after combining from all CPUs). You can viz a horizontal slab, vertical profile at a point, and transect in animation mode.  SCHISM的可视化工具(已经落后了)

xmgr5: a matlab-like 2D plot tool.

You can find the ACE package in the source code bundle; there is an install_notes inside.

## 编译

ACE-tools的源码位于 /schism-5.11.0/src/Utility/ACE

Windows系统下使用Cygwin工具编译ACE-tools, 参考./ACE-Cygwin.md

Ubuntu Linux系统下，参考上述路径下的Install_notes_new

需要安装必须的库：

apt install libgd-dev libmotif-dev libxpm-dev

## ACE-tool使用视频

[ACE-tool使用视频下载链接](http://ccrm.vims.edu/yinglong/wiki_files/SELFE_Tutorial_May2012.flv)
