# NetCDF安装摘记(包括c库和fortran库） 

## 1、安装环境

系统环境：Ubuntu 18.04+bash

编译器：gcc 4.8.5

C库版本：netcdf-4.6.1.tar.gz

Fortran库版本：netcdf-fortran-4.4.4.tar.gz

 
## 2、C库安装(without netCDF4 and HDF5support)

(1)设定环境变量
```
export CC=gcc
export CPPFLAGS='-DNDEBUG –DgFortran'
export FC=gfortran
export FFLAGS='-O -w'
export CXX=g++
```
(2)安装C库(安装至/opt/netcdf)
```
tar -xzvfnetcdf-4.6.1.tar.gz
cd netcdf-4.6.1
./configure --disable-netcdf-4 --disable-dap --prefix=/opt/netcdf
make
make check
make install
```
 
## 3、Fortran库安装(安装至/opt/netcdf)
```
tar -xzvf netcdf-fortran-4.4.4.tar.gz
cd netcdf-fortran-4.4.4
CPPFLAGS=-I/opt/netcdf/include LD_LIBRARY_PATH=/opt/netcdf/lib:${LD_LIBRARY_PATH} LDFLAGS=-L/opt/netcdf/lib 
./configure --prefix=/opt/netcdf
make
make check
make install
```