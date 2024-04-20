#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash pnetcdf-1.12.2.sh buildpath installpath"
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

wget -t 0 -c -P $buildpath https://parallel-netcdf.github.io/Release/pnetcdf-1.12.2.tar.gz
tar xf $buildpath/pnetcdf-1.12.2.tar.gz -C $buildpath && rm -rf $buildpath/pnetcdf-1.12.2.tar.gz
cd $buildpath/pnetcdf-1.12.2

CC=mpicc CXX=mpicxx FC=mpifort F77=mpifort CFLAGS="-fPIC -DPIC" FCFLAGS="-fPIC" FFLAGS="-fPIC" CXXFLAGS="-fPIC -DPIC" SEQ_CC=mpicc \
./configure --prefix=$installpath/pnetcdf-1.12.2 --enable-shared
make -j$(nproc) && make -j$(nproc) install

cat>"$installpath/pnetcdf-1.12.2/pnetcdf_modulefiles"<<EOF
#%Module1.0
conflict pnetcdf
variable modfile [file normalize [info script]]
proc getModulefileDir {} {
    variable modfile
    set modfile_path [file dirname \$modfile]
    return \$modfile_path
}
set pwd [getModulefileDir]
set PNETCDF \$pwd
setenv PNETCDF \$PNETCDF 
prepend-path PATH \$PNETCDF/bin
prepend-path LD_LIBRARY_PATH \$PNETCDF/lib
EOF

rm -rf $buildpath/pnetcdf-1.12.2

