#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash hmpi-1.1.1.sh buildpath installpath"
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
yum install -y wget tar perl-Data-Dumper autoconf automake libtool numactl binutils systemd-devel valgrind flex environment-modules gcc-c++ git

download_pkg()
{
    wget -t 0 -c -P $buildpath $1
    tar xf $buildpath/v1.1.1-huawei.tar.gz -C $buildpath
    rm -rf $buildpath/v1.1.1-huawei.tar.gz
}

download_pkg https://github.com/kunpengcompute/hmpi/archive/refs/tags/v1.1.1-huawei.tar.gz
download_pkg https://github.com/kunpengcompute/hucx/archive/refs/tags/v1.1.1-huawei.tar.gz
download_pkg https://github.com/kunpengcompute/xucg/archive/refs/tags/v1.1.1-huawei.tar.gz
cp -r $buildpath/xucg-1.1.1-huawei/* $buildpath/hucx-1.1.1-huawei/src/ucg/
cd $buildpath/hucx-1.1.1-huawei
git init && ./autogen.sh
./contrib/configure-opt --prefix=$installpath/hmpi-1.1.1/hucx-1.1.1-huawei --disable-numa CC=clang CXX=clang++ FC=flang
for file in `find . -name Makefile`;do sed -i "s/-Werror//g" $file;done
for file in `find . -name Makefile`;do sed -i "s/-implicit-function-declaration//g" $file;done
make -j$(nproc)
make -j$(nproc) install
export PATH=$installpath/hmpi-1.1.1/hucx-1.1.1-huawei/bin:$PATH
export LD_LIBRARY_PATH=$installpath/hmpi-1.1.1/hucx-1.1.1-huawei/lib:$LD_LIBRARY_PATH
export INCLUDE=$installpath/hmpi-1.1.1/hucx-1.1.1-huawei/include:$INCLUDE
cd $buildpath/hmpi-1.1.1-huawei
./autogen.pl CC=clang CXX=clang++ FC=flang
# We use multiple thread in openmpi here for async HPX and HDF5 afterwards.   --enable-mpi-thread-multiple
./configure --prefix=$installpath/hmpi-1.1.1/hmpi-1.1.1-huawei --with-platform=contrib/platform/mellanox/optimized --enable-mpil-compatibility \
	--with-ucx=$installpath/hmpi-1.1.1/hucx-1.1.1-huawei --enable-mpi-thread-multiple CC=clang CXX=clang++ FC=flang
make -j$(nproc)
make -j$(nproc) install
cat>"$installpath/hmpi-1.1.1/hmpi_modulefiles"<<EOF
#%Module1.0
conflict hmpi
variable modfile [file normalize [info script]]
proc getModulefileDir {} {
    variable modfile
    set modfile_path [file dirname \$modfile]
    return \$modfile_path
}
set pwd [getModulefileDir]
set HUCX \$pwd/hucx-1.1.1-huawei
setenv HUCX \$HUCX
prepend-path PATH \$HUCX/bin
prepend-path LD_LIBRARY_PATH \$HUCX/lib
prepend-path INCLUDE \$HUCX/include
set HMPI \$pwd/hmpi-1.1.1-huawei
setenv HMPI \$HMPI
prepend-path PATH \$HMPI/bin
prepend-path LD_LIBRARY_PATH \$HMPI/lib
prepend-path INCLUDE \$HMPI/include
EOF

rm -rf $buildpath/xucg-1.1.1-huawei $buildpath/hucx-1.1.1-huawei $buildpath/hmpi-1.1.1-huawei 
