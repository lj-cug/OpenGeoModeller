#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash wrf-4.2.sh buildpath installpath"
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
yum install -y  environment-modules csh time patch libatomic perl-Data-Dumper autoconf automake libtool numactl binutils systemd-devel valgrind flex wget

set -e
wget -t 0 -c -P $buildpath https://github.com/wrf-model/WRF/archive/refs/tags/v4.2.tar.gz
tar xf $buildpath/v4.2.tar.gz -C $buildpath && rm -rf $buildpath/v4.2.tar.gz
cp configure.defaults.patch $buildpath/WRF-4.2
cd $buildpath/WRF-4.2
patch -p1 < ./configure.defaults.patch
ln -fs /usr/lib64/libz.so.1.2.11 /usr/lib64/libz.so
echo 4 | ./configure
sed -i 's/derf/erf/g' ./phys/module_mp_SBM_polar_radar.F
sed -i -e 's/gcc/clang/g' \
	-e 's/gfortran/flang/g' \
	-e 's/mpicc/mpicc -DMPI2_SUPPORT/g' \
	-e 's/ -ftree-vectorize//g' \
	-e 's/length-none/length-0/g' \
	-e 's/-frecord-marker\=4/ /g' \
	-e 's/\-w \-O3 \-c/-mcpu=native \-w \-O3 \-c/g' \
	-e 's/\# \-g $(FCNOOPT) .*/\-g/g' \
	-e 's/$(FCBASEOPTS_NO_G)/-mcpu=native $(OMP) $(FCBASEOPTS_NO_G)/g' configure.wrf
./compile -j $(nproc) em_real | tee compile.log
mkdir -p $installpath/wrf-4.2 && cp -r $buildpath/WRF-4.2/* $installpath/wrf-4.2

rm -rf $buildpath/WRF-4.2

