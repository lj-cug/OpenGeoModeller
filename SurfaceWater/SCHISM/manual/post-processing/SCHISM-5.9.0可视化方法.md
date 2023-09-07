# SCHISM-5.9.0以后版本的可视化

SCHISM-5.9.0以后版本，使用NetCDF格式保存计算结果，因此可以使用visIT等开源可视化软件做后处理，更方便

一些用户编写的[可视化Matlab和Python脚本](http://ccrm.vims.edu/w/index.php/Share_your_tools#Python_scripts_to_visualize_SCHISM_outputs)，可更灵活地后处理计算结果

[schismview可视化工具](./schismview-main)

## 使用Matlab可视化

The directory Utility/Vis_Matlab/ has matlab scripts that can visualize outputs along a horizontal slab (at a fixed z level or at a sigma level) or vertical transects. In particular, SCHISM_SLAB2.m and SCHISM_TRANSECT2.m for for the new scribed outputs, while SCHISM_SLAB.m and SCHISM_TRANSECT.m are for the old outputs (schout*.nc).

## 使用Python可视化

[more coming] You can also find several packages on the [Forum site](http://ccrm.vims.edu/w/index.php/Share_your_tools)

## 使用VisIt可视化

The most comprehensive way to visualize SCHISM nc4 outputs is via VisIt.

Shortly after v5.9.0, we have successfully implemented a new mode of efficient I/O using dedicated 'scribes'. At runtime, the user needs to specify number of scribe cores (= # of 3D outputs variables (vectors counted as 2) plus 1), and the code, compiled without OLDIO, will output combined netcdf outputs for each 3D variable and also all 2D variables in out2d*.nc. Sample 3D outputs are: salinity_*.nc, horizontalVelX_*.nc etc - note that vectors variable names end with X,Y. You can find sample outputs [here](http://ccrm.vims.edu/yinglong/SVN_large_files/Scribe_IO_outputs/). Sample outputs using OLDIO (schout*.nc) can be found [here](http://ccrm.vims.edu/yinglong/SVN_large_files/SCHISM_v5.6.1_sample_outputs/).

You can download newer versions of VisIt plugins c/o Jon Shu, DWR by following these steps:

### On Windows 7 or 10

First download VisIt from LLNL site and install. Use default dir (and record it), e.g. C:\Users\username\AppData\Local\Programs\LLNL\VisIt*
Make sure MS visualc++ 2012 x64 is installed. If not, google it and install and restart (this is required for using multi-core VisIt)
Download pre-built plug-in, developed at California Dept of Water Resource

For VisIt v2.13.3
For VisIt v3.1.4
For VisIt v3.3.1

You need to put dlls to: Documents/VisIt/databases (create new folders if necessary), except netcdf_c++.dll. The NetCDF DLL needs to be copied to %USERPROFILE%\LLNL\VisIt 3.3.1 or an equivalent VisIt installation directory.

After these steps, you'd be able to read in SCHISM outputs in ViSIt; look for SCHISM, gr3 format from the dropdown list. To load in vectors, select only the X file.

### On Linux Systems

Newer versions can be found at the master branch of [github](./schism_visit_plugin-master).

Note:

Note that the new plugins also work with the old I/O (combined schout*.nc) or even the older binary outputs. To visualize any variables under new I/O with VisIt, you'll always need corresponding out2d*.nc; additionally for any 3D variables, VisIt also needs corresponding zCoordinates*.nc.
