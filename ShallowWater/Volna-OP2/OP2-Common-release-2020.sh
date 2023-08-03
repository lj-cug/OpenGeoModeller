#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash OP2-Common.sh buildpath installpath"
    echo ":param buildpath: 应用构建绝对路径"
    echo ":param installpath: 应用安装绝对路径"
    exit 1
fi


buildpath=$1
installpath=$2
[[ "$buildpath" =~ .*/$ ]] && buildpath=${buildpath%/*}
[[ "$installpath" =~ .*/$ ]] && installpath=${installpath%/*}
[[ "$buildpath" == "$installpath" ]] && echo "构建路径与安装路径需为不同路径" && exit 1
set -e

cp Makefile_OP2.patch $buildpath
cp OP2-Common-release-2020.tar.xz $buildpath
cd $buildpath
#git clone -b release/2020 https://github.com/OP-DSL/OP2-Common.git
tar xf OP2-Common-release-2020.tar.xz 

# Setup the temporatory environmental variables for OP2
CC=`which mpicc`
CXX=`which mpicxx`
FC=`which mpif90`
export MPI_INSTALL_PATH=/workspace/public/software/mpi/hmpi/1.1.1/bisheng2.1.0/ompi
export CUDA_INSTALL_PATH=/usr/local/cuda-11.4
export PTSCOTCH_INSTALL_PATH=$installpath/scotch-6.0.6
export PARMETIS_INSTALL_PATH=$installpath/parmetis-4.0.3/metis
export HDF5_INSTALL_PATH=$installpath/hdf5-1.13.0
export OP2_COMPILER='clang'
export LD_LIBRARY_PATH=$CUDA_INSTALL_PATH/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PARMETIS_INSTALL_PATH/lib:$PTSCOTCH_INSTALL_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HDF5_INSTALL_PATH/lib:$LD_LIBRARY_PATH
export NV_ARCH='Ampere'

cp $buildpath/Makefile_OP2.patch $buildpath/OP2-Common-release-2020/op2/c
cd OP2-Common-release-2020/op2/c
patch -p0 Makefile Makefile_OP2.patch
make

#rm -rf $buildpath/OP2-Common
