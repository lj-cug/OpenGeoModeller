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

get_parmetis_integrate()
{
cnt=1
number=10
while [ $cnt -lt $number ]
do
    set +e
    echo "正在下载ParMETIS源码包，请耐心等待......"
    wget -t 1 -c -P $buildpath http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz
    result_code=$?
    if [ $result_code -ne 0 ];then
        cnt=$[$cnt+1]
    else
        echo "下载ParMETIS源码包成功"
        break
    fi
done

if [ $cnt -eq $number ];then
    echo "下载ParMETIS源码包失败,将dependency/src目录下的离线包拷贝并解压到构建目录下"
    cp dependency/src/parmetis-4.0.3.tar.gz  $buildpath/parmetis-4.0.3.tar.gz
    tar -xvf $buildpath/parmetis-4.0.3.tar.gz -C $buildpath && rm -rf $buildpath/parmetis-4.0.3.tar.gz
fi
set -e 
}

buildpath=$1
installpath=$2
[[ "$buildpath" =~ .*/$ ]] && buildpath=${buildpath%/*}
[[ "$installpath" =~ .*/$ ]] && installpath=${installpath%/*}
[[ "$buildpath" == "$installpath" ]] && echo "构建路径与安装路径需为不同路径" && exit 1
set -e
yum install -y wget tar environment-modules

get_parmetis_integrate

cd $buildpath/parmetis-4.0.3
sed -i -e 's/\#define IDXTYPEWIDTH 32/\#define IDXTYPEWIDTH 64/g' metis/include/metis.h
yum install cmake -y
cd metis
make config shared=1 prefix=$installpath/parmetis-4.0.3/metis
cd ../
sed -i -e '29i add_compile_options(-fPIC)' CMakeLists.txt
make config shared=1 prefix=$installpath/parmetis-4.0.3/metis
make install
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
