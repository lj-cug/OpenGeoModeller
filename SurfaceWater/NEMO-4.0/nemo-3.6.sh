#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash nemo-3.6.sh buildpath installpath"
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
yum install -y environment-modules patch svn
set -e

mkdir -p $buildpath/NEMO
cp arch-aarch64_gnu.fcm $buildpath/NEMO/
cp namelist_cfg.patch $buildpath/NEMO/
cd $buildpath/NEMO
echo "export NETCDF_DIR=$installpath/netcdf" >> nemo-env.sh
echo "export HDF_DIR=$installpath/hdf5-1.12.1" >> nemo-env.sh
echo "export XIOS_DIR=$installpath/xios-1.0" >> nemo-env.sh
source nemo-env.sh
svn co http://forge.ipsl.jussieu.fr/nemo/svn/NEMO/releases/release-3.6/NEMOGCM
cd NEMOGCM
cp ../arch-aarch64_gnu.fcm ./ARCH/arch-aarch64_gnu.fcm 
cd ./CONFIG/
./makenemo -m aarch64_gnu -j$(nproc) -r GYRE -n 'MY_GYRE' add_key "key_nosignedzero"
cp $buildpath/NEMO/namelist_cfg.patch ./MY_GYRE/EXP00/
cd ./MY_GYRE/EXP00/
patch -p0 namelist_cfg namelist_cfg.patch
mkdir -p $installpath/nemo-3.6 && cp -r $buildpath/NEMO/NEMOGCM/* $installpath/nemo-3.6
cat>"$installpath/nemo-3.6/nemo_modulefiles"<<EOF
#%Module1.0
conflict nemo
variable modfile [file normalize [info script]]
proc getModulefileDir {} {
    variable modfile
    set modfile_path [file dirname \$modfile]
    return \$modfile_path
}
set pwd [getModulefileDir]
set NEMO \$pwd
setenv NEMO \$NEMO
prepend-path PATH \$NEMO/CONFIG/MY_GYRE/BLD/bin
EOF

rm -rf $buildpath/NEMO
