# Windows��Cygwin��ʹ��NCView

1. ʹ��cygwin��setup.exe��װ NetCDF, HDF5, Curl, libXaw, libICE, udunits, libexpat �� libpng��

������ѡ������������"netcdf", "hdf5", "curl", "Xaw", "libICE", "udunits" "libexpat" and "libpng"

������װ���¹��߰�:
```
��������NetCDF 4.2.1.1-1: libnetcdf-devel, libnetcdf7, netcdf
��������HDF5 1.8.9-1: hdf5, libhdf5-devel, libhdf5_7
��������curl 7.27.0-1: libcurl4
��������libXaw 1.0.11-1: libXaw-devel, libXaw7
��������libICE 1.0.8-1: libICE-devel, libICE6
��������libpng 1.5.12-1: libpng-devel, libpng15, libpng
��������udunits 2.1.24-1: libudunits-devel, libudunits0, udunits
��������libexpat 2.1.0-1: libexpat1, libexpat1-devel
```

2.���ز���װNCView��
```
����>wget ftp://cirrus.ucsd.edu/pub/ncview/ncview-2.1.1.tar.gz ������ֱ�����������󣬵���Ӧ��Ŀ¼�½�ѹҲ���ԣ�
����>tar xvfz ncview-2.1.1.tar.gz
����>cd ncview-2.1.1
����>./configure --prefix=/home/rsignell
����>make install
```

 ��װ�ɹ����� /home/rsignell/bin/ncview���ҵ�������ִ���ļ�