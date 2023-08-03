#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash mgcfd-1.0.0.sh buildpath installpath"
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

cp Makefile_MGCFD.patch $buildpath
cp MG-CFD-app-OP2-master.zip $buildpath
cd $buildpath
# We use master branch due to the heavy development of MG-CFD-app-OP2
# git clone https://github.com/warwick-hpsc/MG-CFD-app-OP2.git
unzip MG-CFD-app-OP2-master.zip
cp $buildpath/Makefile_MGCFD.patch $buildpath/MG-CFD-app-OP2-master

# Set envrionmental variables building for MG-CFD-app-OP2
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
export OP2_INSTALL_PATH=$buildpath/OP2-Common-release-2020/op2/c
export CUDA_LIB=$CUDA_INSTALL_PATH/lib64
export CC=`which clang`
export CXX=`which clang++`
export FC=`which flang`

cd $buildpath/MG-CFD-app-OP2-master
patch -p0 Makefile Makefile_MGCFD.patch

make seq
make openmp
make mpi
make mpi_openmp
make mpi_vec
make cuda
make mpi_cuda
mkdir -p $installpath/MG-CFD-app-OP2 && cp -r $buildpath/MG-CFD-app-OP2-master/* $installpath/MG-CFD-app-OP2
cat>"$installpath/MG-CFD-app-OP2/MG-CFD-app-OP2_modulefiles"<<EOF
#%Module1.0
conflict mgcfd
variable modfile [file normalize [info script]]
proc getModulefileDir {} {
    variable modfile
    set modfile_path [file dirname \$modfile]
    return \$modfile_path
}
set pwd [getModulefileDir]
set MGCFD \$pwd
setenv MGCFD \$MGCFD
prepend-path PATH \$MGCFD/bin
EOF

rm -rf $buildpath/MG-CFD-app-OP2-master
