#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash build.sh buildpath installpath"
    echo ":param buildpath: 应用构建绝对路径"
    echo ":param installpath: 应用安装绝对路径"
    exit 1
fi

[[ ! "$1" =~ ^/.* || "$1" = "/" ]] &&  echo "请输入正确的构建路径" && exit 1
[[ ! "$2" =~ ^/.* || "$2" = "/" ]] &&  echo "请输入正确的安装路径" && exit 1

create_dir()
{
    local path=$1
    [[ "$path" =~ .*/$ ]] && path=${path%/*}
    if [[ ! -d "$path" && ! -f "$path" ]]; then mkdir -p $path; else path=$path`date "+%y%m%d%H%M%S"` && mkdir -p $path; fi
    echo $path
}

buildpath=$(create_dir $1)
installpath=$(create_dir $2)

yum install -y m4 environment-modules systemd-devel zlib*
source /etc/profile
module purge
set -e
dep_dir="dependency"
bash $dep_dir/bisheng-compiler-2.1.0.sh $buildpath $installpath
module use $installpath/bisheng-compiler-2.1.0
module load $installpath/bisheng-compiler-2.1.0/bisheng_modulefiles

bash $dep_dir/hmpi-1.1.1.sh $buildpath $installpath
module use $installpath/hmpi-1.1.1
module load $installpath/hmpi-1.1.1/hmpi_modulefiles

bash $dep_dir/zlib-1.2.11.sh $buildpath $installpath
module use $installpath/zlib-1.2.11
module load $installpath/zlib-1.2.11/zlib_modulefiles

bash $dep_dir/hdf5-1.12.1.sh $buildpath $installpath
module use $installpath/hdf5-1.12.1
module load $installpath/hdf5-1.12.1/hdf5_modulefiles

bash $dep_dir/netcdf-c-4.8.1.sh $buildpath $installpath
module use $installpath/netcdf
module load $installpath/netcdf/netcdf_modulefiles

bash $dep_dir/netcdf-fortran-4.5.3.sh $buildpath $installpath
module use $installpath/netcdf
module load $installpath/netcdf/netcdf_modulefiles

bash $dep_dir/parmetis-4.0.3.sh $buildpath $installpath
module use $installpath/parmetis-4.0.3
module load $installpath/parmetis-4.0.3/parmetis_modulefiles

bash ww3-6.07.1.sh $buildpath $installpath
mkdir -p pkg/ww3-6.07.1-hpc && cp -r $installpath/* pkg/ww3-6.07.1-hpc/
cat>"pkg/ww3-6.07.1-hpc/run.sh"<<EOF
#!/bin/bash
sudo yum install -y \
  m4 environment-modules systemd-devel patch autoconf \
  automake libtool python python-devel python-setuptools \
  python2-pip wget tar libatomic gcc-c++ zlib zlib-devel

source /etc/profile

# Get the absolute path of the current script
current_dir="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd -P)"

for f in "\${current_dir}"/*; do
  if [[ -d "\${f}" && -f "\$(ls "\${f}"/*_modulefiles 2>&1)" ]]; then
    module use "\${f}" && module load "\${f}"/*_modulefiles
  fi
done

[[ ! -d "$installpath" ]] && mkdir -p $installpath && cp -r \$current_dir/* $installpath/

source env_ww3.sh
export ww3_dir=$installpath/WW3-6.07.1/model

echo -e "\033[1;32;1mWAVEWATCH-III environment initialization completed.\n\033[0m"
EOF
cd pkg/ && tar zcvf cesm-2.1.1-hpc.tar.gz cesm-2.1.1-hpc

rm -rf $buildpath $installpath
