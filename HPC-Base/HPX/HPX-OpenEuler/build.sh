#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash build.sh buildpath installpath"
    echo ":param buildpath: 应用构建绝对路径"
    echo ":param installpath: 应用安装绝对路径"
    exit 1
fi


buildpath=$1
installpath=$2
[[ "$buildpath" =~ .*/$ ]] && buildpath=${buildpath%/*}
[[ "$installpath" =~ .*/$ ]] && installpath=${installpath%/*}

set -e

dep_dir="dependency"

module use /workspace/public/software/modules
module load compilers/bisheng/2.1.0/bisheng2.1.0
module use /workspace/public/software/modules
module load mpi/hmpi/1.1.1/bisheng2.1.0
module use /workspace/public/software/modules
module load compilers/cuda/11.14.1

bash $dep_dir/asio-1.21.0.sh $buildpath $installpath
module use $installpath/asio-1.21.0
module load $installpath/asio-1.21.0/asio_modulefiles

bash $dep_dir/gperftools-2.6.1.sh $buildpath $installpath
module use $installpath/gperftools-2.6.1
module load $installpath/gperftools-2.6.1/gperftools_modulefiles

bash $dep_dir/hwloc-2.7.1.sh $buildpath $installpath
module use $installpath/hwloc-2.7.1
module load $installpath/hwloc-2.7.1/hwloc_modulefiles

bash $dep_dir/boost-1.75.0.sh $buildpath $installpath
module use $installpath/boost-1.75.0
module load $installpath/boost-1.75.0/boost_modulefiles

#hpx-1.8.1 is the newest stable version
bash hpx-1.8.1.sh $buildpath $installpath
module use $installpath/hpx-1.8.1
module load $installpath/hpx-1.8.1/hpx_modulefiles

# If you want to use HPXCL-0.1, HPX-1.6.0 must be installed.
# build hpx-1.6.0, lowest requirement for 3rd parties, and default-used for hpxcl-1.6.0
#bash hpx-1.6.0 $buildpath $installpath
#module use $installpath/hpx-1.6.0
#module load $installpath/hpx-1.6.0/hpx_modulefiles

# hpxcl-0.1-alpha should be based on hpx-1.6.0
#bash hpxcl-0.1-alpha $buildpath $installpath
#module use $installpath/hpxcl-0.1
#module load $installpath/hpxcl-0.1/hpxcl_modulefiles

