#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash openblas-0.3.6.sh buildpath installpath"
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

wget -t 10 -c -P $buildpath https://github.com/xianyi/OpenBLAS/archive/refs/tags/v0.3.6.tar.gz



make -j$(nproc)
make install

cat>"$installpath/openblas-0.3.6/openblas_modulefiles"<<EOF
#%Module1.0
conflict openblas
variable modfile [file normalize [info script]]
proc getModulefileDir {} {
    variable modfile
    set modfile_path [file dirname \$modfile]
    return \$modfile_path
}
set pwd [getModulefileDir]
set OPENBLAS \$pwd
setenv OPENBLAS \$OPENBLAS
prepend-path LD_LIBRARY_PATH \$OPENBLAS/lib
prepend-path INCLUDE \$OPENBLAS/include
EOF

rm -rf $buildpath/OpenBLAS-0.3.6

