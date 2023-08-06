#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash netcdf-fortran-4.4.1.sh buildpath installpath"
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

wget -t 0 -c -P $buildpath https://downloads.unidata.ucar.edu/netcdf-fortran/4.4.1/netcdf-fortran-4.4.1.tar.gz 
cp libtool.patch $buildpath/libtool.patch
tar xf $buildpath/netcdf-fortran-4.4.1.tar.gz -C $buildpath && rm -rf $buildpath/netcdf-fortran-4.4.1.tar.gz
cd $buildpath/netcdf-fortran-4.4.1

./configure --prefix=$installpath/netcdf CPPFLAGS="-I$installpath/hdf5-1.12.1/include -I$installpath/netcdf/include" LDFLAGS="-L$installpath/hdf5-1.12.1/lib -L$installpath/netcdf/lib" --build=aarch64-unknown-linux-gnu --enable-static=yes --enable-shared CFLAGS="-O3 -fPIC -Wno-incompatible-pointer-types-discards-qualifiers -Wno-non-literal-conversion" FCFLAGS="-O3 -fPIC" LDFLAGS="-Wl" CC=mpicc FC=mpif90 CXX=mpicxx
cp $buildpath/libtool.patch ./libtool.patch
patch -p0 libtool libtool.patch
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
prepend-path LD_LIBRARY_PATH \$NETCDF/lib
prepend-path INCLUDE \$NETCDF/include
EOF

rm -rf $buildpath/netcdf-fortran-4.4.1
