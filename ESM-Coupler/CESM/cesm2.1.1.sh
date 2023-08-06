#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash cesm2.1.1.sh buildpath installpath"
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
yum install -y wget svn environment-modules
set -e

wget -t 10 -c -P $buildpath https://github.com/ESCOMP/CESM/archive/refs/tags/release-cesm2.1.1.tar.gz
tar xf $buildpath/release-cesm2.1.1.tar.gz -C $buildpath && rm -rf $buildpath/release-cesm2.1.1.tar.gz
# Prepare the patch files
cp nf_mod.F90.patch $buildpath/CESM-release-cesm2.1.1
cp ionf_mod.F90.patch $buildpath/CESM-release-cesm2.1.1
cp pionfput_mod.F90.in.patch $buildpath/CESM-release-cesm2.1.1
cp pionfwrite_mod.F90.in.patch $buildpath/CESM-release-cesm2.1.1
cp shr_sys_mod.F90.patch $buildpath/CESM-release-cesm2.1.1
cp config_machines.xml.patch $buildpath/CESM-release-cesm2.1.1
cp config_compilers.xml.patch $buildpath/CESM-release-cesm2.1.1
cd $buildpath/CESM-release-cesm2.1.1

echo 'Downloading cime...'
wget -t 10 -c https://github.com/ESMCI/cime/archive/refs/tags/cime5.6.19.tar.gz
tar xf cime5.6.19.tar.gz && mv cime-cime5.6.19 cime && rm -f cime5.6.19.tar.gz

echo 'Downloading components...'
mkdir -p components
cd components
# cam
svn co https://svn-ccsm-models.cgd.ucar.edu/cam1/release_tags/cam_cesm2_1_rel_29/components/cam/
#cice
wget -t 10 -c https://github.com/ESCOMP/CESM_CICE5/archive/refs/tags/cice5_20190321.tar.gz
tar xf cice5_20190321.tar.gz && mv CESM_CICE5-cice5_20190321 cice && rm -f cice5_20190321.tar.gz
#cism
wget -t 10 -c https://github.com/ESCOMP/CISM-wrapper/archive/refs/tags/release-cesm2.0.04.tar.gz
tar xf release-cesm2.0.04.tar.gz && mv CISM-wrapper-release-cesm2.0.04 cism && rm -f release-cesm2.0.04.tar.gz
cd cism
wget -t 10 -c https://github.com/ESCOMP/cism/archive/refs/tags/release-cism2.1.03.tar.gz
tar xf release-cism2.1.03.tar.gz && mv CISM-release-cism2.1.03 source_cism && rm -f release-cism2.1.03.tar.gz
cd ..
#clm
wget -t 10 -c https://github.com/ESCOMP/CTSM/archive/refs/tags/release-clm5.0.25.tar.gz
tar xf release-clm5.0.25.tar.gz && mv CTSM-release-clm5.0.25 clm && rm -f release-clm5.0.25.tar.gz
cd clm
wget -t 10 -c https://github.com/ncar/fates-release/archive/refs/tags/fates_s1.21.0_a7.0.0_br_rev2.tar.gz
tar xf fates_s1.21.0_a7.0.0_br_rev2.tar.gz && mv fates-release-fates_s1.21.0_a7.0.0_br_rev2 src/fates && rm -f fates_s1.21.0_a7.0.0_br_rev2.tar.gz
wget -t 10 -c https://github.com/ESCOMP/ptclm/archive/refs/tags/PTCLM2_180611.tar.gz
tar xf PTCLM2_180611.tar.gz && mv PTCLM-PTCLM2_180611 tools/PTCLM && rm -f PTCLM2_180611.tar.gz
cd ..
#mosart
wget -t 10 -c https://github.com/ESCOMP/mosart/archive/refs/tags/release-cesm2.0.03.tar.gz
tar xf release-cesm2.0.03.tar.gz && mv MOSART-release-cesm2.0.03 mosart && rm -f release-cesm2.0.03.tar.gz
#pop
wget -t 10 -c https://github.com/ESCOMP/POP2-CESM/archive/refs/tags/pop2_cesm2_1_rel_n06.tar.gz
tar xf pop2_cesm2_1_rel_n06.tar.gz && mv POP2-CESM-pop2_cesm2_1_rel_n06 pop && rm -f pop2_cesm2_1_rel_n06.tar.gz
cd pop
wget -t 10 -c https://github.com/CVMix/CVMix-src/archive/refs/tags/v0.93-beta.tar.gz
tar xf v0.93-beta.tar.gz && mv CVMix-src-0.93-beta externals/CVMix && rm -f v0.93-beta.tar.gz
wget -t 10 -c https://github.com/marbl-ecosys/MARBL/archive/refs/tags/cesm2.1-n00.tar.gz
tar xf cesm2.1-n00.tar.gz && mv MARBL-cesm2.1-n00 externals/MARBL && rm -f cesm2.1-n00.tar.gz
cd ..
#rtm
wget -t 10 -c https://github.com/ESCOMP/rtm/archive/refs/tags/release-cesm2.0.02.tar.gz
tar xf release-cesm2.0.02.tar.gz && mv RTM-release-cesm2.0.02 rtm && rm -f release-cesm2.0.02.tar.gz
#ww3
wget -t 10 -c https://github.com/ESCOMP/WW3-CESM/archive/refs/tags/ww3_cesm2_1_rel_01.tar.gz
tar xf ww3_cesm2_1_rel_01.tar.gz && mv WW3-CESM-ww3_cesm2_1_rel_01 ww3 && rm -f ww3_cesm2_1_rel_01.tar.gz
cd ..

# Patch some file for compiling CESM using Bisheng-Compiler 2.1
echo 'Config CESM cime and components...'
cd $buildpath/CESM-release-cesm2.1.1
./manage_externals/checkout_externals -S
cp nf_mod.F90.patch ./cime/src/externals/pio1/pio
cp ionf_mod.F90.patch ./cime/src/externals/pio1/pio
cp pionfput_mod.F90.in.patch ./cime/src/externals/pio1/pio
cp pionfwrite_mod.F90.in.patch ./cime/src/externals/pio1/pio
cp shr_sys_mod.F90.patch ./cime/src/share/util
cp config_machines.xml.patch ./cime/config/cesm/machines
cp config_compilers.xml.patch ./cime/config/cesm/machines
echo 'Patching related files...'
cd $buildpath/CESM-release-cesm2.1.1/cime/src/externals/pio1/pio
patch -p0 nf_mod.F90 nf_mod.F90.patch
patch -p0 ionf_mod.F90 ionf_mod.F90.patch
patch -p0 pionfput_mod.F90.in pionfput_mod.F90.in.patch
patch -p0 pionfwrite_mod.F90.in pionfwrite_mod.F90.in.patch
cd $buildpath/CESM-release-cesm2.1.1/cime/src/share/util
patch -p0 shr_sys_mod.F90 shr_sys_mod.F90.patch
cd $buildpath/CESM-release-cesm2.1.1/cime/config/cesm/machines
hostname openEuler
patch -p0 config_machines.xml config_machines.xml.patch
patch -p0 config_compilers.xml config_compilers.xml.patch
cd $buildpath/CESM-release-cesm2.1.1
mkdir -p $installpath/cesm-2.1.1 && cp -r $buildpath/CESM-release-cesm2.1.1/* $installpath/cesm-2.1.1
rm -rf $buildpath/CESM-release-cesm2.1.1
