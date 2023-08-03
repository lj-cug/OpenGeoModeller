#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash gperftools-2.6.1.sh buildpath installpath"
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

wget -t 10 -c -P $buildpath https://github.com/gperftools/gperftools/archive/gperftools-2.6.1.tar.gz
tar xf $buildpath/gperftools-2.6.1.tar.gz -C $buildpath && rm -rf $buildpath/gperftools-2.6.1.tar.gz
cd $buildpath/gperftools-gperftools-2.6.1
./autogen.sh
CC=`which gcc`
CXX=`which g++`
./configure --prefix=$installpath/gperftools-2.6.1
make -j$(nproc)
make install
cat>"$installpath/gperftools-2.6.1/gperftools_modulefiles"<<EOF
#%Module1.0
conflict gperftools
variable modfile [file normalize [info script]]
proc getModulefileDir {} {
    variable modfile
    set modfile_path [file dirname \$modfile]
    return \$modfile_path
}
set pwd [getModulefileDir]
set GPERFTOOLS \$pwd
setenv GPERFTOOLS \$GPERFTOOLS
prepend-path LD_LIBRARY_PATH \$GPERFTOOLS/lib
prepend-path INCLUDE \$GPERFTOOLS/include
EOF

rm -rf $buildpath/asio-asio-1-21-0
