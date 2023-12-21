#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash hdf5-1.12.1.sh buildpath installpath"
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

wget -t 0 -c -P $buildpath https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.1/src/hdf5-1.12.1.tar.gz
tar xf $buildpath/hdf5-1.12.1.tar.gz -C $buildpath && rm -rf $buildpath/hdf5-1.12.1.tar.gz
cd $buildpath/hdf5-1.12.1
CC=mpicc CXX=mpicxx FC=mpifort F77=mpi77 CFLAGS="-fPIC -DPIC" FCFLAGS="-fPIC -DPIC" FFLAGS="-fPIC" \
./configure --prefix=$installpath/hdf5-1.12.1 --with-zlib=$installpath/zlib-1.2.11 --enable-shared --enable-fortran --enable-static --enable-parallel
sed -i 's@\\\$wl-soname@-install_name@g' libtool
make -j$(nproc) && make -j$(nproc) install
cat>"$installpath/hdf5-1.12.1/hdf5_modulefiles"<<EOF
#%Module1.0
conflict hdf5
variable modfile [file normalize [info script]]
proc getModulefileDir {} {
    variable modfile
    set modfile_path [file dirname \$modfile]
    return \$modfile_path
}
set pwd [getModulefileDir]
set HDF5 \$pwd
setenv HDF5 \$HDF5
prepend-path LD_LIBRARY_PATH \$HDF5/lib
EOF

rm -rf $buildpath/hdf5-1.12.1

