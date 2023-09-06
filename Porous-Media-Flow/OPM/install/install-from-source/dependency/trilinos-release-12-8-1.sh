#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash trilinos-release-12-8-1.sh buildpath installpath"
    echo ":param buildpath: Ӧ�ù�������·��"
    echo ":param installpath: Ӧ�ð�װ����·��"
    exit 1
fi

[[ ! "$1" =~ ^/.* || ! -d "$1" || "$1" = "/" ]] &&  echo "��������ȷ�Ĺ���·��" && exit 1
[[ ! "$2" =~ ^/.* || ! -d "$2" || "$2" = "/" ]] &&  echo "��������ȷ�İ�װ·��" && exit 1

buildpath=$1
installpath=$2
[[ "$buildpath" =~ .*/$ ]] && buildpath=${buildpath%/*}
[[ "$installpath" =~ .*/$ ]] && installpath=${installpath%/*}
[[ "$buildpath" == "$installpath" ]] && echo "����·���밲װ·����Ϊ��ͬ·��" && exit 1
set -e

git clone https://github.com/trilinos/Trilinos.git
cd Trilinos
git checkout trilinos-release-12-8-1
mkdir build
(
  cd build
  cmake \
    -D CMAKE_INSTALL_PREFIX=$installpath \
    -D TPL_ENABLE_MPI:BOOL=ON \
    -D MPI_BASE_DIR:PATH=/usr/local \
    -D Trilinos_ENABLE_ALL_PACKAGES:BOOL=OFF \
    -D Trilinos_ENABLE_Zoltan:BOOL=ON \
    ../
  make -j $(nproc)
)

cat>"$installpath/zoltan-12.8.1/zoltan_modulefiles"<<EOF
#%Module1.0
conflict dune
variable modfile [file normalize [info script]]
proc getModulefileDir {} {
    variable modfile
    set modfile_path [file dirname \$modfile]
    return \$modfile_path
}
set pwd [getModulefileDir]
set DUNE \$pwd
setenv DUNE \$DUNE
prepend-path LD_LIBRARY_PATH \$DUNE/lib
prepend-path INCLUDE \$DUNE/include
EOF

rm -rf $buildpath/Trilinos