#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash hwloc-2.7.1.sh buildpath installpath"
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
yum install -y wget tar environment-modules

wget -t 10 -c -P $buildpath https://download.open-mpi.org/release/hwloc/v2.7/hwloc-2.7.1.tar.gz
tar xf $buildpath/hwloc-2.7.1.tar.gz -C $buildpath && rm -rf $buildpath/hwloc-2.7.1.tar.gz
cd $buildpath/hwloc-2.7.1
./autogen.sh
CC=`which clang`
CXX=`which clang++`
./configure --prefix=$installpath/hwloc-2.7.1
make -j$(nproc)
make install
cat>"$installpath/hwloc-2.7.1/hwloc_modulefiles"<<EOF
#%Module1.0
conflict hwloc
variable modfile [file normalize [info script]]
proc getModulefileDir {} {
    variable modfile
    set modfile_path [file dirname \$modfile]
    return \$modfile_path
}
set pwd [getModulefileDir]
set HWLOC \$pwd
setenv HWLOC \$HWLOC
prepend-path LD_LIBRARY_PATH \$HWLOC/lib
prepend-path INCLUDE \$HWLOC/include
EOF

rm -rf $buildpath/hwloc-2.7.1

