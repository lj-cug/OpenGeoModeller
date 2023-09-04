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

apt-get install -y environment-modules
source /etc/profile
module purge
set -e

dep_dir="dependency"

# Use openmpi, gnu compiler in Ubuntu-20.04

# Install prerequisites for OPM: (1) from PPA repository; (2) Build from source code
# (1) from PPA repository
bash $dep_dir/PPA_prerequisites_Ubuntu.sh

# (2) Build dune-2.9 from source code
#bash $dep_dir/dune-2.9.sh $buildpath $installpath
#module use $installpath/dune-2.9
#module load $installpath/dune-2.9/dune_modulefiles

# Build Zoltan fro source code
# bash $dep_dir/trilinos-release-12-8-1.sh $buildpath $installpath
#module use $installpath/zoltan
#module load $installpath/zoltan/zoltan_modulefiles

bash opm-2019-Nov.sh $buildpath $installpath

mkdir -p pkg/opm-2019-Nov-hpc && cp -r $installpath/* pkg/opm-2019-Nov-hpc/
cat>"pkg/opm-2019-Nov-hpc/run.sh"<<EOF
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

cd \$current_dir
[[ ! -d "$installpath/hmpi-1.1.1" ]] && mkdir -p $installpath/hmpi-1.1.1 && cp -r \$current_dir/hmpi-1.1.1/* $installpath/hmpi-1.1.1/

echo -e "\033[1;32;1mOPM environment initialization completed.\n\033[0m"
EOF
cd pkg/ && tar zcvf opm-2019-Nov-hpc.tar.gz opm-2019-Nov-hpc

rm -rf $buildpath $installpath
