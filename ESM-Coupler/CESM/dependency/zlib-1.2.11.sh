#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash zlib-1.12.11.sh buildpath installpath"
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

wget -t 10 -c -P $buildpath http://www.zlib.net/fossils/zlib-1.2.11.tar.gz
tar xf $buildpath/zlib-1.2.11.tar.gz -C $buildpath && rm -rf $buildpath/zlib-1.2.11.tar.gz
cd $buildpath/zlib-1.2.11
CC=clang ./configure --prefix=$installpath/zlib-1.2.11
make -j$(nproc) && make -j$(nproc) install
cat>"$installpath/zlib-1.2.11/zlib_modulefiles"<<EOF
#%Module1.0
conflict zlib
variable modfile [file normalize [info script]]
proc getModulefileDir {} {
    variable modfile
    set modfile_path [file dirname \$modfile]
    return \$modfile_path
}
set pwd [getModulefileDir]
set ZLIB \$pwd
setenv ZLIB \$ZLIB
prepend-path LD_LIBRARY_PATH \$ZLIB/lib
EOF

rm -rf $buildpath/zlib-1.2.11
