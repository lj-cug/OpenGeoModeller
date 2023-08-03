#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash asio-1.21.0.sh buildpath installpath"
    echo ":param buildpath: 应用构建绝对路径"
    echo ":param installpath: 应用安装绝对路径"
    exit 1
fi


buildpath=$1
installpath=$2
[[ "$buildpath" =~ .*/$ ]] && buildpath=${buildpath%/*}
[[ "$installpath" =~ .*/$ ]] && installpath=${installpath%/*}
[[ "$buildpath" == "$installpath" ]] && echo "构建路径与安装路径需为不同路径" && exit 1

set -e

cp memory.patch $buildpath
#wget -t 10 -c -P $buildpath https://github.com/chriskohlhoff/asio/archive/refs/tags/asio-1-21-0.tar.gz
cp /workspace/home/migration/zhusihai/install_libs/asio-1-21-0.tar.gz $buildpath

tar xf $buildpath/asio-1-21-0.tar.gz -C $buildpath && rm -rf $buildpath/asio-1-21-0.tar.gz
cd $buildpath/asio-asio-1-21-0/asio
./autogen.sh
CC=`which clang`
CXX=`which clang++`
./configure --prefix=$installpath/asio-1.21.0
make -j$(nproc)
make install
# for building HPX-1.8.1
cp $buildpath/memory.patch $installpath/asio-1.21.0/include/asio/detail
cd $installpath/asio-1.21.0/include/asio/detail
patch -p0 memory.hpp memory.patch
cat>"$installpath/asio-1.21.0/asio_modulefiles"<<EOF
#%Module1.0
conflict asio
variable modfile [file normalize [info script]]
proc getModulefileDir {} {
    variable modfile
    set modfile_path [file dirname \$modfile]
    return \$modfile_path
}
set pwd [getModulefileDir]
set ASIO \$pwd
setenv ASIO \$ASIO
prepend-path LD_LIBRARY_PATH \$ASIO/lib
EOF

rm -rf $buildpath/asio-asio-1-21-0
