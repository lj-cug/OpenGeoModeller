#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash ww3-6.07.1.sh buildpath installpath"
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
yum install -y  environment-modules csh time patch libatomic autoconf automake libtool numactl binutils systemd-devel valgrind flex wget gcc gcc-gfortran
set -e

wget -t 10 -c -P $installpath https://github.com/NOAA-EMC/WW3/archive/refs/tags/6.07.1.tar.gz
tar xf $installpath/6.07.1.tar.gz -C $installpath && rm -f $installpath/6.07.1.tar.gz
cp env_ww3.sh  $installpath/WW3-6.07.1/model/bin/
cp cmplr.env.patch $installpath/WW3-6.07.1/model/bin/
cp comp.Gnu.patch $installpath/WW3-6.07.1/model/bin/
cp link.Gnu.patch $installpath/WW3-6.07.1/model/bin/
cp w3_setup.patch $installpath/WW3-6.07.1/model/bin/
cd $installpath/WW3-6.07.1/model/bin
source env_ww3.sh
patch -p0 w3_setup w3_setup.patch
cd $installpath/WW3-6.07.1
./model/bin/w3_setup model
cd ./model/bin
patch -p0 cmplr.env cmplr.env.patch
patch -p0 comp.Gnu comp.Gnu.patch
patch -p0 link.Gnu link.Gnu.patch
export ww3_dir=$installpath/WW3-6.07.1/model
./w3_clean -c
./w3_setup $ww3_dir -c Gnu -s UKMO
./w3_new
./w3_make ww3_grid
