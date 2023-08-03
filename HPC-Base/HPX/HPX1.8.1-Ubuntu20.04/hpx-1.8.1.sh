#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash hpx-1.8.1.sh buildpath installpath"
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

wget -t 10 -c -P $buildpath https://github.com/STEllAR-GROUP/hpx/archive/refs/tags/1.8.1.tar.gz
tar -xvf $buildpath/1.8.1.tar.gz -C $buildpath && rm -rf $buildpath/1.8.1.tar.gz
cd $buildpath/hpx-1.8.1
mkdir build
cd build

# functions-ON: async-mpi and async-cuda 
# functions-OFF: examples, and We turn off HPX_WITH_DYNAMIC_HPX_MAIN for use "HPX_main()" building by makefile
cmake \
-DCMAKE_CXX_COMPILER=clang++ \
-DCMAKE_CXX_COMPILER_AR=llvm-ar \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_CXX_FLAGS_RELEASE=-O2 \
-DCMAKE_INSTALL_PREFIX=$installpath/hpx1.8.1 \
-DBoost_INCLUDE_DIR=$installpath/boost-1.75.0/include \
-DASIO_INCLUDE_DIR=$installpath/asio-1.21.0/include \
-DHPX_WITH_MALLOC=tcmalloc \
-DHWLOC_INCLUDE_DIR=$installpath/hwloc-2.7.1/include \
-DHWLOC_LIBRARY=$installpath/hwloc-2.7.1/lib/libhwloc.so \
-DHPX_WITH_GENERIC_CONTEXT_COROUTINES=ON \
-DHPX_WITH_ASYNC_MPI=ON \
-DHPX_WITH_CUDA=ON \
-DTCMALLOC_LIBRARY=$installpath/gperftools-2.6.1/lib/libtcmalloc.so \
-DTCMALLOC_INCLUDE_DIR=$installpath/gperftools-2.6.1/include \
-DHPX_WITH_MAX_CPU_COUNT=256 \
-DHPX_WITH_DYNAMIC_HPX_MAIN=OFF \
-DHPX_WITH_EXAMPLES=OFF \
..
make -j$(nproc) && make install
cat>"$installpath/hpx-1.8.1/hpx_modulefiles"<<EOF
#%Module1.0
conflict hpx
variable modfile [file normalize [info script]]
proc getModulefileDir {} {
    variable modfile
    set modfile_path [file dirname \$modfile]
    return \$modfile_path
}
set pwd [getModulefileDir]
set HPX \$pwd
setenv HPX \$HPX
prepend-path LD_LIBRARY_PATH \$HPX/lib
prepend-path INCLUDE \$HPX/include
EOF

rm -rf $buildpath/hpx-1.8.1

