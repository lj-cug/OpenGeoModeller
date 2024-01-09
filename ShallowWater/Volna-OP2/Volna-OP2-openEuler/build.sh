#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash build.sh buildpath installpath"
    echo ":param buildpath: 应用构建绝对路径"
    echo ":param installpath: 应用安装绝对路径"
    exit 1
fi

[[ ! "$1" =~ ^/.* || "$1" = "/" ]] &&  echo "请输入正确的构建路径" && exit 1
[[ ! "$2" =~ ^/.* || "$2" = "/" ]] &&  echo "请输入正确的安装路径" && exit 1

# create path from input arguments
echo "create path from input arguments"
buildpath=$1
installpath=$2
[[ "$buildpath" =~ .*/$ ]] && buildpath=${buildpath%/*}
[[ "$installpath" =~ .*/$ ]] && installpath=${installpath%/*}

set -e

dep_dir="dependency"

#bash $dep_dir/bisheng-compiler-2.1.0.sh $buildpath $installpath
#bash $dep_dir/hmpi-1.1.1.sh $buildpath $installpath
# or
# module load building environments on logging node
module use /workspace/public/software/modules
module load compilers/bisheng/2.1.0/bisheng2.1.0
module use /workspace/public/software/modules
module load mpi/hmpi/1.1.1/bisheng2.1.0
module use /workspace/public/software/modules
module load compilers/cuda/11.14.1

bash $dep_dir/zlib-1.2.11.sh $buildpath $installpath
module use $installpath/zlib-1.2.11
module load $installpath/zlib-1.2.11/zlib_modulefiles

# build HDF5-1.13.0
bash $dep_dir/hdf5-1.13.0.sh $buildpath $installpath
module use $installpath/hdf5-1.13.0
module load $installpath/hdf5-1.13.0/hdf5_modulefiles

bash $dep_dir/parmetis-4.0.3.sh $buildpath $installpath
module use $installpath/parmetis-4.0.3
module load $installpath/parmetis-4.0.3/parmetis_modulefiles

bash $dep_dir/scotch-6.0.6.sh $buildpath $installpath

bash OP2-Common-release-2020.sh $buildpath $installpath

# Build MGCFD-OP2 based on OP2-Common-2020
bash MGCFD-OP2-master.sh $buildpath $installpath
mkdir -p pkg/MGCFD-OP2-master-hpc && cp -r $installpath/* pkg/MGCFD-OP2-master-hpc/

rm -rf $buildpath

