#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash hdf5-1.12.1.sh buildpath installpath"
    echo ":param buildpath: 应用构建绝对路径"
    echo ":param installpath: 应用安装绝对路径"
    exit 1
fi


buildpath=$1
installpath=$2
[[ "$buildpath" =~ .*/$ ]] && buildpath=${buildpath%/*}
[[ "$installpath" =~ .*/$ ]] && installpath=${installpath%/*}
set -e

wget -t 10 -c -P $buildpath https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_13_0.tar.gz
tar xf $buildpath/hdf5-1_13_0.tar.gz -C $buildpath && rm -rf $buildpath/hdf5-1_13_0.tar.gz
cd $buildpath/hdf5-hdf5-1_13_0
./autogen.sh
CC=`which mpicc`
CXX=`which mpicxx`
./configure --prefix=$installpath/hdf5-1.13.0 --enable-shared --enable-static --enable-parallel --enable-threadsafe --enable-unsupported --enable-file-locking=no
make -j$(nproc) && make install

cat>"$installpath/hdf5-1.13.0/hdf5_modulefiles"<<EOF
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
prepend-path PATH \$HDF5/bin
prepend-path LD_LIBRARY_PATH \$HDF5/lib
EOF

rm -rf $buildpath/hdf5-hdf5-1_13_0
