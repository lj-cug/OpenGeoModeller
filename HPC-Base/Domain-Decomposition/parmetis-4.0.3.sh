#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash parmetis-4.0.3.sh buildpath installpath"
    echo ":param buildpath: 应用构建绝对路径"
    echo ":param installpath: 应用安装绝对路径"
    exit 1
fi

[[ ! "$1" =~ ^/.* || ! -d "$1" || "$1" = "/" ]] &&  echo "请输入正确的构建路径" && exit 1
[[ ! "$2" =~ ^/.* || ! -d "$2" || "$2" = "/" ]] &&  echo "请输入正确的安装路径" && exit 1
if [[ "$1" == "$2" ]]; then echo "构建路径与安装路径不能相同，请输入正确的路径"; exit 1; fi

buildpath=$1
installpath=$2
[[ "$buildpath" =~ .*/$ ]] && buildpath=${buildpath%/*}
[[ "$installpath" =~ .*/$ ]] && installpath=${installpath%/*}
set -e
yum install -y wget tar environment-modules cmake

#wget -t 10 -c -P $buildpath http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz

export MY_PATH=/workspace/home/migration/zhusihai

cp parmetis-4.0.3.tar.gz $MY_PATH
tar xf $MY_PATH/parmetis-4.0.3.tar.gz -C $MY_PATH && rm -rf $MY_PATH/parmetis-4.0.3.tar.gz
cd $MY_PATH/parmetis-4.0.3
export CC=`which clang`
export CXX=`which clang++`
export FC=`which flang`
sed -i -e 's/\#define IDXTYPEWIDTH 32/\#define IDXTYPEWIDTH 64/g' metis/include/metis.h
cd metis
make config shared=1 prefix=$MY_PATH/parmetis-4.0.3/metis
make -j$(nproc) install   # build metis
cd ../
sed -i -e '29i add_compile_options(-fPIC)' CMakeLists.txt
make config shared=1 prefix=$installpath/parmetis-4.0.3/metis
make -j$(nproc) install   # build parmetis

cat>"$installpath/parmetis-4.0.3/parmetis_modulefiles"<<EOF
#%Module1.0
conflict parmetis
variable modfile [file normalize [info script]]
proc getModulefileDir {} {
    variable modfile
    set modfile_path [file dirname \$modfile]
    return \$modfile_path
}
set pwd [getModulefileDir]
set PARMETIS \$pwd
setenv PARMETIS \$PARMETIS
prepend-path LD_LIBRARY_PATH \$PARMETIS/lib
prepend-path INCLUDE \$PARMETIS/include
EOF

rm -rf $buildpath/parmetis-4.0.3
