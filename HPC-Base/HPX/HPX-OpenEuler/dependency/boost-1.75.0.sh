#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash boost-1.75.0.sh buildpath installpath"
    echo ":param buildpath: 应用构建绝对路径"
    echo ":param installpath: 应用安装绝对路径"
    exit 1
fi

buildpath=$1
installpath=$2
[[ "$buildpath" =~ .*/$ ]] && buildpath=${buildpath%/*}
[[ "$installpath" =~ .*/$ ]] && installpath=${installpath%/*}

set -e

#wget -t 10 -c -P $buildpath https://boostorg.jfrog.io/artifactory/main/release/1.75.0/source/boost_1_75_0.tar.gz
cp /workspace/home/migration/zhusihai/install_libs/boost_1_75_0.tar.gz $buildpath

tar -xvf $buildpath/boost_1_75_0.tar.gz  -C $buildpath && rm -rf $buildpath/boost_1_75_0.tar.gz
cd $buildpath/boost_1_75_0
./bootstrap.sh --with-toolset=clang
./b2 install --prefix=$installpath/boost-1.75.0
cat>"$installpath/boost-1.75.0/boost_modulefiles"<<EOF
#%Module1.0
conflict boost
variable modfile [file normalize [info script]]
proc getModulefileDir {} {
    variable modfile
    set modfile_path [file dirname \$modfile]
    return \$modfile_path
}
set pwd [getModulefileDir]
set BOOST \$pwd
setenv BOOST \$BOOST
prepend-path LD_LIBRARY_PATH \$BOOST/lib
prepend-path INCLUDE \$BOOST/include
EOF

rm -rf $buildpath/boost_1_75_0

