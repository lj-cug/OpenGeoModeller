#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash blitz-1.0.1.sh buildpath installpath"
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

wget -t 10 -c -P $buildpath https://github.com/blitzpp/blitz/archive/refs/tags/1.0.1.tar.gz
tar xf $buildpath/1.0.1.tar.gz -C $buildpath && rm -rf $buildpath/1.0.1.tar.gz
cd $buildpath/blitz-1.0.1
autoreconf -fiv
./configure --prefix=$installpath/blitz-1.0.1 --enable-fortran --enable-64bit
make lib
make install
cp -rf ./src $installpath/blitz-1.0.1
cat>"$installpath/blitz-1.0.1/blitz_modulefiles"<<EOF
#%Module1.0
conflict blitz
variable modfile [file normalize [info script]]
proc getModulefileDir {} {
    variable modfile
    set modfile_path [file dirname \$modfile]
    return \$modfile_path
}
set pwd [getModulefileDir]
set BLITZ \$pwd
setenv BLITZ \$BLITZ
prepend-path LD_LIBRARY_PATH \$BLITZ/lib
prepend-path INCLUDE \$BLITZ/include
prepend-path BLITZ_INCLUDEDIR \$BLITZ/include
EOF

rm -rf $buildpath/blitz-1.0.1
