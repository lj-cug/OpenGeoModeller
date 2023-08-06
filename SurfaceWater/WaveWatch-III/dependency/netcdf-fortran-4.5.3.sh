#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash netcdf-fortran-4.5.3.sh buildpath installpath"
    echo ":param buildpath: 应用构建绝对路径"
    echo ":param installpath: 应用安装绝对路径"
    exit 1
fi

[[ ! "$1" =~ ^/.* || ! -d "$1" || "$1" = "/" ]] &&  echo "请输入正确的构建路径" && exit 1
[[ ! "$2" =~ ^/.* || ! -d "$2" || "$2" = "/" ]] &&  echo "请输入正确的安装路径" && exit 1

buildpath=$1
installpath=$2
[[ "$buildpath" =~ .*/$ ]] && buildpath=${buildpath%/*}
[[ "$installpath" =~ .*/$ ]] && installpath=${installpath%/*}
[[ "$buildpath" == "$installpath" ]] && echo "构建路径与安装路径需为不同路径" && exit 1
set -e
yum install -y wget tar environment-modules

wget -t 10 -c -P $buildpath https://downloads.unidata.ucar.edu/netcdf-fortran/4.5.3/netcdf-fortran-4.5.3.tar.gz 
tar xf $buildpath/netcdf-fortran-4.5.3.tar.gz -C $buildpath && rm -rf $buildpath/netcdf-fortran-4.5.3.tar.gz
cd $buildpath/netcdf-fortran-4.5.3
CC=mpicc CXX=mpicxx FC=mpifort F77=mpifort CPPFLAGS="-I$installpath/hdf5-1.12.1/include -I$installpath/netcdf/include" \
CFLAGS="-I$installpath/hdf5-1.12.1/include -I$installpath/netcdf/include -fPIC -DPIC" \
FCFLAGS="-I$installpath/hdf5-1.12.1/include -I$installpath/netcdf/include -fPIC -DPIC" \
LDFLAGS="-L$installpath/hdf5-1.12.1/lib -L$installpath/netcdf/lib" \
./configure --prefix=$installpath/netcdf --enable-shared --enable-dap --enable-pic --disable-doxygen --enable-static --enable-largefile
sed -i \
	-e 's/\\\$wl-soname/-install_name/g' \
	-e 's/\\\$wl--whole-archive/-Wl,-force-load,/g' \
	-e 's/\\\$wl--no-whole-archive//g' libtool
make -j$(nproc) && make -j$(nproc) install

rm -rf $buildpath/netcdf-fortran-4.5.3
