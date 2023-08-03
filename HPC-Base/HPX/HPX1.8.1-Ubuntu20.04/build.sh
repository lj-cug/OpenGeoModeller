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

create_dir()
{
    local path=$1
    [[ "$path" =~ .*/$ ]] && path=${path%/*}
    if [[ ! -d "$path" && ! -f "$path" ]]; then mkdir -p $path; else path=$path`date "+%y%m%d%H%M%S"` && mkdir -p $path; fi
    echo $path
}

# create random pathname
#buildpath=$(create_dir $1)
#installpath=$(create_dir $2)

# create path from input arguments
buildpath=$1
installpath=$2
[[ "$buildpath" =~ .*/$ ]] && buildpath=${buildpath%/*}
[[ "$installpath" =~ .*/$ ]] && installpath=${installpath%/*}

apt install -y m4 environment-modules time patch libatomic autoconf automake libtool numactl binutils systemd-devel wget
source /etc/profile
module purge
set -e

dep_dir="dependency"

# we should build hmpi with mpi_thread_multiple
bash $dep_dir/hmpi-1.1.1.sh $buildpath $installpath
module use $installpath/hmpi-1.1.1
module load $installpath/hmpi-1.1.1/hmpi_modulefiles

# Build the 3rd parties used for hpx-1.8.1
wget https://github.com/Kitware/CMake/releases/download/v3.23.3/cmake-3.23.3-linux-aarch64.tar.gz
tar xf cmake-3.23.3-linux-aarch64.tar.gz
export CMAKE_HOME=$PWD/cmake-3.23.3
export PATH=$CMAKE_HOME/bin:$PATH

# Then, build asio-1-21-0
bash $dep_dir/asio-1.21.0.sh $buildpath $installpath
module use $installpath/asio-1.21.0
module load $installpath/asio-1.21.0/asio_modulefiles

# Then, gperftools-2.6.1
bash $dep_dir/gperftools-2.6.1.sh $buildpath $installpath
module use $installpath/gperftools-2.6.1
module load $installpath/gperftools-2.6.1/gperftools_modulefiles

# Then, hwloc-2.7.1
bash $dep_dir/hwloc-2.7.1.sh $buildpath $installpath
module use $installpath/hwloc-2.7.1
module load $installpath/hwloc-2.7.1/hwloc_modulefiles

# Then, boost-1.75.0
bash $dep_dir/hboost-1.75.0.sh $buildpath $installpath
module use $installpath/boost-1.75.0
module load $installpath/boost-1.75.0/boost_modulefiles

# hpx-1.8.1 is the newest stable version, which can use async-mpi and async-cuda.
bash hpx-1.8.1.sh $buildpath $installpath
module use $installpath/hpx-1.8.1
module load $installpath/hpx-1.8.1/hpx_modulefiles
# or
# build hpx-1.6.0, version 1.6.0 is stable, low requirement for 3rd parties, and suitable for hpxcl-1.6.0
#bash hpx-1.6.0 $buildpath $installpath
#module use $installpath/hpx-1.6.0
#module load $installpath/hpx-1.6.0/hpx_modulefiles

# Then, hpxcl-0.1-alpha should be based on hpx-1.6.0
#bash hpxcl-0.1-alpha $buildpath $installpath
#module use $installpath/hpxcl-0.1
#module load $installpath/hpxcl-0.1/hpxcl_modulefiles

