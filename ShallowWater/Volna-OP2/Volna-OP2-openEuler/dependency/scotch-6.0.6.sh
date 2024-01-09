#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash scotch-6.0.6.sh buildpath installpath"
    echo ":param buildpath: 应用构建绝对路径"
    echo ":param installpath: 应用安装绝对路径"
    exit 1
fi


buildpath=$1
installpath=$2
[[ "$buildpath" =~ .*/$ ]] && buildpath=${buildpath%/*}
[[ "$installpath" =~ .*/$ ]] && installpath=${installpath%/*}
set -e

wget -c -t 10 -P $buildpath https://gitlab.inria.fr/scotch/scotch/-/archive/v6.0.6/scotch-v6.0.6.tar.gz
tar xf $buildpath/scotch-v6.0.6.tar.gz -C $buildpath && rm -rf $buildpath/scotch-v6.0.6.tar.gz
cp Makefile.inc.patch $buildpath/

cd $buildpath/scotch-v6.0.6/src
cp Make.inc/Makefile.inc.x86-64_pc_linux2 Makefile.inc
cp $buildpath/Makefile.inc.patch ./
patch -p0 Makefile.inc Makefile.inc.patch
make libscotch
make libptscotch
mkdir -p $installpath/scotch-6.0.6
# 直接安装SCOTCH到安装目录
make prefix=$installpath/scotch-6.0.6 install
rm -rf $buildpath/scotch_6.0.6
