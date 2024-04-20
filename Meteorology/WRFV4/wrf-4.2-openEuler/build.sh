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

yum install -y m4 environment-modules systemd-devel
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

bash $dep_dir/pnetcdf-1.12.2.sh $buildpath $installpath
module use $installpath/pnetcdf-1.12.2
module load $installpath/pnetcdf-1.12.2/pnetcdf_modulefiles

bash $dep_dir/netcdf-c-4.8.1.sh $buildpath $installpath
module use $installpath/netcdf
module load $installpath/netcdf/netcdf_modulefiles

bash $dep_dir/netcdf-fortran-4.5.3.sh $buildpath $installpath
module use $installpath/netcdf
module load $installpath/netcdf/netcdf_modulefiles

bash wrf-4.2.sh $buildpath $installpath

mkdir -p pkg/wrf-4.2-hpc && cp -r $installpath/* pkg/wrf-4.2-hpc/
cat>"pkg/wrf-4.2-hpc/run.sh"<<EOF
#!/bin/bash
yum install -y m4 environment-modules systemd-devel csh time patch libatomic perl-Data-Dumper \\
autoconf automake libtool numactl binutils valgrind flex wget tar libatomic gcc-c++ git
source /etc/profile &> /dev/null 
current_dir=\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" &> /dev/null && pwd)
for d in \$(ls \$current_dir)
do
    [[ ! \$d =~ "wrf" && ! -f "\$current_dir/\$d" ]] && cd \$current_dir/\$d && module use \$(pwd) && module load \$(pwd)/*modulefiles
done
cd \$current_dir
[[ ! -d "$installpath/hdf5-1.12.1/lib/" ]] && {
mkdir -p $installpath/hdf5-1.12.1/lib/; \
ln -sf \$current_dir/hdf5-1.12.1/lib/libhdf5_hl.so $installpath/hdf5-1.12.1/lib/; \
ln -sf \$current_dir/hdf5-1.12.1/lib/libhdf5.so $installpath/hdf5-1.12.1/lib/
}
[[ ! -d "$installpath/hmpi-1.1.1" ]] && mkdir -p $installpath/hmpi-1.1.1 && cp -r \$current_dir/hmpi-1.1.1/* $installpath/hmpi-1.1.1/
EOF
cd pkg/ && tar zcvf wrf-4.2-hpc.tar.gz wrf-4.2-hpc

rm -rf $buildpath $installpath
