#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "The format is as follows:"
    echo "bash dune-2.9.sh buildpath installpath"
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
set -e

# Set of modules to build
modules="dune-common dune-geometry dune-grid dune-istl"

# Clone modules, check out the 2.9 release.
for m in $modules; do
    echo "==================================================="
    echo "=        Cloning module: $m"
    echo "==================================================="
    (
        if [ ! -d "$m" ]; then
            git clone -b releases/2.9 https://gitlab.dune-project.org/core/$m.git
        else
            echo "******** Skipping $m, module already cloned."
        fi
    )
done

# Build the modules, and install them to the chosen directory
for m in $modules; do
    echo "==================================================="
    echo "=        Building module: $m"
    echo "==================================================="
    (
        cd $m
        builddir="build-cmake"
        if [ ! -d "$builddir" ]; then
            mkdir "$builddir"
            cd "$builddir"
            cmake -DCMAKE_INSTALL_PREFIX=$installpath ".."
            make -j $(nproc)
            make install
        else
            echo "******** Skipping $m, build dir $builddir already exists."
        fi
    )
done

cat>"$installpath/dune-2.9/dune_modulefiles"<<EOF
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

for m in $modules; do
    (
rm -rf $buildpath/$m	
    )
done
