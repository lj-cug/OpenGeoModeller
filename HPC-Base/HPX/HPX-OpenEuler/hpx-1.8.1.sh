#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash hpx-1.8.1.sh buildpath installpath"
    echo ":param buildpath: 应用构建绝对路径"
    echo ":param installpath: 应用安装绝对路径"
    exit 1
fi

buildpath=$1
installpath=$2
[[ "$buildpath" =~ .*/$ ]] && buildpath=${buildpath%/*}
[[ "$installpath" =~ .*/$ ]] && installpath=${installpath%/*}
[[ "$buildpath" == "$installpath" ]] && echo "构建路径与安装路径需为不同路径" && exit 1
#yum install -y  environment-modules time patch libatomic autoconf automake libtool numactl binutils systemd-devel wget
set -e


# We need CMAKE-3.23 to build hpx-1.8.1, but in OpenEuler-20.03, CMAKE version is 3.19
#wget -t 10 -c -P $buildpath https://github.com/Kitware/CMake/releases/download/v3.23.3/cmake-3.23.3-linux-aarch64.tar.gz
cp /workspace/home/migration/zhusihai/install_libs/cmake-3.23.3-linux-aarch64.tar.gz $buildpath

tar -xvf $buildpath/cmake-3.23.3-linux-aarch64.tar.gz -C $installpath
rm -rf $buildpath/cmake-3.23.3-linux-aarch64.tar.gz
export PATH=$installpath/cmake-3.23.3-linux-aarch64/bin:$PATH


#wget -t 10 -c -P $buildpath https://github.com/STEllAR-GROUP/hpx/archive/refs/tags/1.8.1.tar.gz
cp /workspace/home/migration/zhusihai/install_libs/1.8.1.tar.gz $buildpath

tar -xvf $buildpath/1.8.1.tar.gz -C $buildpath && rm -rf $buildpath/1.8.1.tar.gz
cd $buildpath/hpx-1.8.1
mkdir -p build
cd build

# functions-OFF: async-mpi and async-cuda 
# functions-OFF: examples
# We turn off HPX_WITH_DYNAMIC_HPX_MAIN for use "HPX_main()" building by makefile
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
-DHPX_WITH_ASYNC_MPI=OFF \
-DHPX_WITH_CUDA=OFF \
-DTCMALLOC_LIBRARY=$installpath/gperftools-2.6.1/lib/libtcmalloc.so \
-DTCMALLOC_INCLUDE_DIR=$installpath/gperftools-2.6.1/include \
-DHPX_WITH_MAX_CPU_COUNT=256 \
-DHPX_WITH_MAX_NUMA_DOMAIN_COUNT=4 \
-DHPX_WITH_DYNAMIC_HPX_MAIN=OFF \
-DHPX_WITH_EXAMPLES=OFF \
-DHPX_WITH_PKGCONFIG=ON \
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

