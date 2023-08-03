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
if [[ "$1" == "$2" ]]; then echo "构建路径与安装路径不能相同，请输入正确的路径"; exit 1; fi

buildpath=$1
installpath=$2
[[ "$buildpath" =~ .*/$ ]] && buildpath=${buildpath%/*}
[[ "$installpath" =~ .*/$ ]] && installpath=${installpath%/*}
set -e
#yum install -y wget tar environment-modules

wget -t 10 -c -P $buildpath https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_13_0.tar.gz
tar xf $buildpath/hdf5-1_13_0.tar.gz -C $buildpath && rm -rf $buildpath/hdf5-1_13_0.tar.gz

# download vol-async
wget -t 10 -c -P $buildpath https://github.com/hpc-io/vol-async/archive/refs/tags/v1.2.tar.gz
tar xf $buildpath/v1.2.tar.gz -C $buildpath && rm -rf $buildpath/v1.2.tar.gz

# download argobots
wget -t 10 -c -P $buildpath https://github.com/pmodels/argobots/archive/refs/tags/v1.1.tar.gz
tar xf $buildpath/v1.1.tar.gz -C $buildpath && rm -rf $buildpath/v1.1.tar.gz

# start to build hdf5-1.3.0 with asyncIO function
cd $buildpath
export H5_DIR=$PWD/hdf5-hdf5-1_13_0
export VOL_DIR=$PWD/vol-async
export ARBT_DIR=$PWD/vol-async/argobots

# build HDF5-1.13.0
cd $H5_DIR
./autogen.sh
CC=`which mpicc`
CXX=`which mpicxx`
./configure --prefix=$installpath/hdf5-1.13.0/install --enable-shared --enable-static --enable-parallel --enable-threadsafe --enable-unsupported --enable-file-locking=no
make -j$(nproc) && make install

# Build argobots
cd $ARBT_DIR
./autogen.sh
./configure --prefix=$installpath/argobots-1.1/install
make -j$(nproc) && make install

# Build vol_async
cd $VOL_DIR/src
# Edit "Makefile"
cp Makefile.summit Makefile
patch -p0 Makefile Makefile.patch
make -j$(nproc)
#make install  # Should we do install?

# We test asyncIO in HDF5-1.13.0
# Set environmental variables.
export LD_LIBRARY_PATH=$VOL_DIR/src:$H5_DIR/install/lib:$ARBT_DIR/install/lib:$LD_LIBRARY_PATH
export HDF5_PLUGIN_PATH="$VOL_DIR/src"
export HDF5_VOL_CONNECTOR="async under_vol=0;under_info={}"

# preload dynamic libraries for Ubuntu OS
export LD_PRELOAD=$H5_DIR/install/lib/libhdf5.so
export LD_PRELOAD=$H5_DIR/install/lib/libhdf5_hl.so
export LD_PRELOAD=$ARBT_DIR/install/lib/libabt.so

# Test vol-async
cd $VOL_DIR/test
cp Makefile.summit Makefile
make

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

rm -rf $buildpath/hdf5-hdf5-1_13_0 $buildpath/vol-async

