#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash bisheng-compiler-2.1.0.sh buildpath installpath"
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
yum install -y wget tar environment-modules libatomic

wget -t 0 -c -P $buildpath https://mirrors.huaweicloud.com/kunpeng/archive/compiler/bisheng_compiler/bisheng-compiler-2.1.0-aarch64-linux.tar.gz
tar xf $buildpath/bisheng-compiler-2.1.0-aarch64-linux.tar.gz -C $buildpath && rm -rf $buildpath/bisheng-compiler-2.1.0-aarch64-linux.tar.gz
mkdir -p $installpath/bisheng-compiler-2.1.0 && cp -r $buildpath/bisheng-compiler-2.1.0-aarch64-linux/* $installpath/bisheng-compiler-2.1.0/

cat>"$installpath/bisheng-compiler-2.1.0/bisheng_modulefiles"<<EOF
#%Module1.0
conflict bisheng
variable modfile [file normalize [info script]]
proc getModulefileDir {} {
    variable modfile
    set modfile_path [file dirname \$modfile]
    return \$modfile_path
}
set pwd [getModulefileDir]
set BISHENG \$pwd
setenv BISHENG \$BISHENG
prepend-path PATH \$BISHENG/bin
prepend-path LD_LIBRARY_PATH \$BISHENG/lib
EOF

rm -rf $buildpath/bisheng-compiler-2.1.0-aarch64-linux

