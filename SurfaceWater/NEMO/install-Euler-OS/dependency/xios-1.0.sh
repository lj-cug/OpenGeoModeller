#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash xios-1.0.sh buildpath installpath"
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
yum install -y wget tar environment-modules svn
yum install -y perl-lib* boost-*
set -e

mkdir -p $buildpath/XIOS
cp bld.cfg.patch $buildpath/XIOS/
cp arch-AARCH64_GNU_LINUX.env $buildpath/XIOS/
cp arch-AARCH64_GNU_LINUX.fcm $buildpath/XIOS/
cp arch-AARCH64_GNU_LINUX.path $buildpath/XIOS/
cp ./dependency/xios-1.0.tar.xz $buildpath/XIOS/
cd $buildpath/XIOS
#svn co -r 703 http://forge.ipsl.jussieu.fr/ioserver/svn/XIOS/branchs/xios-1.0 xios-1.0
tar xf xios-1.0.tar.xz
cd xios-1.0
cp $buildpath/XIOS/bld.cfg.patch ./bld.cfg.patch
patch -p0 bld.cfg bld.cfg.patch 
 cp -f ../arch-AARCH64_GNU_LINUX.env ./arch/
cp -f ../arch-AARCH64_GNU_LINUX.fcm ./arch/
cp -f ../arch-AARCH64_GNU_LINUX.path ./arch/
chmod +x ./make_xios
cd extern
ln -sf $installpath/blitz-1.0.1 ./blitz
ln -sf $installpath/netcdf ./netcdf4
ln -sf /usr/include/boost ./boost
cd ..
./make_xios --dev --job $(nproc) --full --arch AARCH64_GNU_LINUX
mkdir -p $installpath/xios-1.0 && cp -r $buildpath/XIOS/xios-1.0/* $installpath/xios-1.0/
cat>"$installpath/xios-1.0/xios_modulefiles"<<EOF
#%Module1.0
conflict xios
variable modfile [file normalize [info script]]
proc getModulefileDir {} {
    variable modfile
    set modfile_path [file dirname \$modfile]
    return \$modfile_path
}
set pwd [getModulefileDir]
set XIOS \$pwd
setenv XIOS \$XIOS
prepend-path LD_LIBRARY_PATH \$XIOS/lib
EOF

rm -rf $buildpath/xios-1.0
