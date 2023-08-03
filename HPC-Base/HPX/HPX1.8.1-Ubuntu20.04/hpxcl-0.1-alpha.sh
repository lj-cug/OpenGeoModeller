#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash hpxcl-0.1-alpha.sh buildpath installpath"
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
yum install -y  environment-modules time patch libatomic autoconf automake libtool numactl binutils systemd-devel wget
set -e

wget -t 10 -c -P $buildpath https://github.com/STEllAR-GROUP/hpxcl/archive/refs/tags/v0.1-alpha.tar.gz
tar -xvf $buildpath/v0.1-alpha.tar.gz -C $buildpath && rm -rf $buildpath/v0.1-alpha.tar.gz
cd $buildpath/hpxcl-0.1
mkdir build
cd build
cmake \
-DCMAKE_CXX_COMPILER=clang++ \
-DCMAKE_INSTALL_PREFIX=$installpath/hpxcl-1.6 \
-DCMAKE_BUILD_TYPE=Release \
-DBoost_INCLUDE_DIR=$installpath/boost-1.75.0/include \
-DBoost_LIBRARY_DIR_RELEASE=$installpath/boost-1.75.0/lib \
-DHPX_DIR=$installpath/hpx-1.6.0/build/lib/cmake/HPX \
-DHPXCL_WITH_CUDA=ON \
-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
-DCUDA_USE_STATIC_CUDA_RUNTIME=ON \
-DCUDA_rt_LIBRARY=/usr/lib64/librt.so \
-HPXCL_CUDA_WITH_STREAM=ON \
-DHPXCL_WITH_BENCHMARK=ON \
..
make -j$(nproc) && make install
cat>"$installpath/hpxcl-0.1/hpxcl_modulefiles"<<EOF
#%Module1.0
conflict hpxcl
variable modfile [file normalize [info script]]
proc getModulefileDir {} {
    variable modfile
    set modfile_path [file dirname \$modfile]
    return \$modfile_path
}
set pwd [getModulefileDir]
set HPXCL \$pwd
setenv HPXCL \$HPXCL
prepend-path LD_LIBRARY_PATH \$HPXCL/lib
prepend-path INCLUDE \$HPXCL/include
EOF

rm -rf $buildpath/hpxcl-0.1

