# Windows下Cygwin中使用NCView

1. 使用cygwin的setup.exe安装 NetCDF, HDF5, Curl, libXaw, libICE, udunits, libexpat 和 libpng：

　　在选择库界面搜索："netcdf", "hdf5", "curl", "Xaw", "libICE", "udunits" "libexpat" and "libpng"

　　安装以下工具包:
```
　　　　NetCDF 4.2.1.1-1: libnetcdf-devel, libnetcdf7, netcdf
　　　　HDF5 1.8.9-1: hdf5, libhdf5-devel, libhdf5_7
　　　　curl 7.27.0-1: libcurl4
　　　　libXaw 1.0.11-1: libXaw-devel, libXaw7
　　　　libICE 1.0.8-1: libICE-devel, libICE6
　　　　libpng 1.5.12-1: libpng-devel, libpng15, libpng
　　　　udunits 2.1.24-1: libudunits-devel, libudunits0, udunits
　　　　libexpat 2.1.0-1: libexpat1, libexpat1-devel
```

2.下载并安装NCView：
```
　　>wget ftp://cirrus.ucsd.edu/pub/ncview/ncview-2.1.1.tar.gz （或者直接下载下来后，到对应的目录下解压也可以）
　　>tar xvfz ncview-2.1.1.tar.gz
　　>cd ncview-2.1.1
　　>./configure --prefix=/home/rsignell
　　>make install
```

 安装成功后在 /home/rsignell/bin/ncview可找到二进制执行文件