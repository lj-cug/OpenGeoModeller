#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash netcdf-c-4.8.1.sh buildpath installpath"
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

wget -t 10 -c -P $buildpath https://downloads.unidata.ucar.edu/netcdf-c/4.8.1/netcdf-c-4.8.1.tar.gz
tar xf $buildpath/netcdf-c-4.8.1.tar.gz -C $buildpath && rm -rf $buildpath/netcdf-c-4.8.1.tar.gz
cd $buildpath/netcdf-c-4.8.1
ln -fs /usr/lib64/libz.so.1.2.11 /usr/lib64/libz.so
CC=mpicc CXX=mpicxx CPPFLAGS="-I$installpath/hdf5-1.12.1/include -I$installpath/zlib-1.2.11/include -I$installpath/pnetcdf-1.12.2/include" \
CFLAGS="-I$installpath/hdf5-1.12.1/include  -I$installpath/zlib-1.2.11/include -I$installpath/pnetcdf-1.12.2/include -fPIC" \
LDFLAGS="-L$installpath/hdf5-1.12.1/lib  -L$installpath/zlib-1.2.11/lib -L$installpath/pnetcdf-1.12.2/lib" \
./configure --prefix=$installpath/netcdf --enable-shared --enable-netcdf-4 --enable-dap --with-pic --disable-doxygen --enable-static --enable-largefile
make -j$(nproc) && make -j$(nproc) install
cat>"$installpath/netcdf/netcdf_modulefiles"<<EOF
#%Module1.0
conflict netcdf
variable modfile [file normalize [info script]]
proc getModulefileDir {} {
    variable modfile
    set modfile_path [file dirname \$modfile]
    return \$modfile_path
}
set pwd [getModulefileDir]
set NETCDF \$pwd
setenv NETCDF \$NETCDF
prepend-path PATH \$NETCDF/bin
prepend-path LD_LIBRARY_PATH \$NETCDF/lib
prepend-path INCLUDE \$NETCDF/include
EOF

rm -rf $buildpath/netcdf-c-4.8.1
