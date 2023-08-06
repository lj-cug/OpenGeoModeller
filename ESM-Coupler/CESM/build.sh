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

bash $dep_dir/openblas-0.3.6.sh $buildpath $installpath
module use $installpath/openblas-0.3.6
module load $installpath/openblas-0.3.6/openblas_modulefiles

bash $dep_dir/hdf5-1.12.1.sh $buildpath $installpath
module use $installpath/hdf5-1.12.1
module load $installpath/hdf5-1.12.1/hdf5_modulefiles

bash $dep_dir/pnetcdf-1.12.2.sh $buildpath $installpath
module use $installpath/pnetcdf-1.12.2
module load $installpath/pnetcdf-1.12.2/pnetcdf_modulefiles

bash $dep_dir/netcdf-c-4.8.1.sh $buildpath $installpath
module use $installpath/netcdf
module load $installpath/netcdf/netcdf_modulefiles

bash $dep_dir/netcdf-fortran-4.4.1.sh $buildpath $installpath
module use $installpath/netcdf
module load $installpath/netcdf/netcdf_modulefiles

bash cesm2.1.1.sh $buildpath $installpath
mkdir -p pkg/cesm-2.1.1-hpc && cp -r $installpath/* pkg/cesm-2.1.1-hpc/
cat>"pkg/cesm-2.1.1-hpc/run.sh"<<EOF
#!/bin/bash
sudo yum install -y \
  m4 environment-modules systemd-devel patch autoconf \
  automake libtool python python-devel python-setuptools \
  python2-pip wget tar libatomic gcc-c++ zlib zlib-devel cmake

source /etc/profile

# Get the absolute path of the current script
current_dir="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd -P)"

for f in "\${current_dir}"/*; do
  if [[ -d "\${f}" && -f "\$(ls "\${f}"/*_modulefiles 2>&1)" ]]; then
    module use "\${f}" && module load "\${f}"/*_modulefiles
  fi
done

cd \$current_dir
[[ ! -d "$installpath/hdf5-1.12.1/lib/" ]] && {
mkdir -p $installpath/hdf5-1.12.1/lib/; \
ln -sf \$current_dir/hdf5-1.12.1/lib/libhdf5_hl.so $installpath/hdf5-1.12.1/lib/; \
ln -sf \$current_dir/hdf5-1.12.1/lib/libhdf5.so $installpath/hdf5-1.12.1/lib/
}

[[ ! -d "$installpath/pnetcdf-1.12.2/lib" ]] && mkdir -p $installpath/pnetcdf-1.12.2/lib && cp -r \$current_dir/pnetcdf-1.12.2/lib/* $installpath/pnetcdf-1.12.2/lib/

[[ ! -d "$installpath/openblas-0.3.6/lib/" ]] && {
mkdir -p $installpath/openblas-0.3.6/lib/; \
ln -sf \$current_dir/openblas-0.3.6/lib/libopenblas.so $installpath/openblas-0.3.6/lib/
}

[[ ! -d "$installpath/zlib-1.2.11/lib/" ]] && {
mkdir -p $installpath/zlib-1.2.11/lib/; \
ln -sf \$current_dir/zlib-1.2.11/lib/libz.so $installpath/zlib-1.2.11/lib/
}

# Re-construct building environment for cesm-2.1.1
[[ ! -d "$installpath/bisheng-compiler-2.1.0" ]] && mkdir -p $installpath/bisheng-compiler-2.1.0 && cp -r \$current_dir/bisheng-compiler-2.1.0/* $installpath/bisheng-compiler-2.1.0/
[[ ! -d "$installpath/hmpi-1.1.1" ]] && mkdir -p $installpath/hmpi-1.1.1 && cp -r \$current_dir/hmpi-1.1.1/* $installpath/hmpi-1.1.1/
[[ ! -d "$installpath/netcdf" ]] && mkdir -p $installpath/netcdf && cp -r \$current_dir/netcdf/* $installpath/netcdf/


export compile_tools_path=$installpath

echo -e "\033[1;32;1mCESM environment initialization completed.\n\033[0m"
EOF
cd pkg/ && tar zcvf cesm-2.1.1-hpc.tar.gz cesm-2.1.1-hpc

rm -rf $buildpath $installpath
