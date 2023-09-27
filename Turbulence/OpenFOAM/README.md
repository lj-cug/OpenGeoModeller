# OpenFOAM安装与使用

[用户手册](https://doc.cfd.direct/openfoam/user-guide-v10/)

# 1. Software for Compilation

## Install packages for repositories and compilation

apt-get install build-essential cmake git ca-certificates

## Install general packages for OpenFOAM

apt-get install flex libfl-dev bison zlib1g-dev libboost-system-dev libboost-thread-dev libopenmpi-dev openmpi-bin gnuplot libreadline-dev libncurses-dev libxt-dev

## Install softwares for paraview

apt-get install libqt5x11extras5-dev libxt-dev qt5-default qttools5-dev curl

# 2. Downloading Source Code

First choose a directory location for the installation directory of OpenFOAM. If the installation is for a single user only, or if the user does not have administrative privileges (or root access) to the machine, we recommend the installation directory is $HOME/OpenFOAM  (i.e. a directory OpenFOAM in the user’s home directory). If the installer has administrator privileges and the installation is for more than one user, one of the ‘standard’ locations can be used, e.g. /usr/local/OpenFOAM, /opt/OpenFOAM, or just /opt.  If necessary, create the chosen installation directory.  
Go into that directory in preparation of cloning repositories.

## Cloning the Repositories

The following instructions are written for the OpenFOAM-dev and ThirdParty-dev repositories.  For the OpenFOAM-11 and ThirdParty-11 repositories follow the instructions substituting the extension (dev → 11) where necessary.

To clone the repositories, execute the following:

git clone https://github.com/OpenFOAM/OpenFOAM-dev.git
git clone https://github.com/OpenFOAM/ThirdParty-dev.git

# 3. Setting the Environment

source $HOME/OpenFOAM/OpenFOAM-dev/etc/bashrc

# 4. Third Party Software

## Installing Scotch/PT-Scotch

OpenFOAM requires Scotch/PT-Scotch version 6 and higher since it includes a fix to allow both the libscotch and libptscotch libraries to be linked to the same application.  Version 6 can be installed as a package for most recent versions of Linux, but not Ubuntu, since the Debian package maintainers have failed to upgrade to version 6 since its release in December 2012.  Note that the packaged version of OpenFOAM-dev for Ubuntu contains the object libraries for Scotch/PT-Scotch v6.0.9 built from ThirdParty-dev, as described next.

If a packaged version is not available, Scotch/PT-Scotch v6.0.9 can be installed simply by going into the ThirdParty directory (e.g. ThirdParty-dev) where the sources can be compiled by running the Allwmake script.

./Allwmake

Note that the Allwmake script is set up to be able to compile other packages, e.g. GCAL, but these packages are ignored unless the source code is downloaded to the ThirdParty directory.

## Installing ParaView

For the current supported version of ParaView , the source code is downloaded automatically. It compiles without modification, except for GCC v10+, when it is automatically patched by the installation script.  ParaView is compiled by running the makeParaView script, i.e.

./makeParaView

Expect ParaView to take a long time to compile, typically several hours on a desktop computer. Following compilation, update the environment by sourcing the .bashrc file as described in 3. Setting the Environment or by typing

wmRefresh

# 5. Compiling OpenFOAM

With the Third Party software installed and environment updated, compile OpenFOAM by going into the OpenFOAM-dev directory and executing the Allwmake script.  Type Allwmake -help for options, but the 2 main choices are to compile in serial with

./Allwmake

or compile in parallel with all available cores/hyperthreads with:

./Allwmake -j

Serial compilation takes several hours, whereas compilation on 8 cores/threads should take approximately one hour, possibly less, depending on the processor.

## Updating the System

The source repositories, in particular OpenFOAM-dev, are updated regularly.  Updates can be “pulled” to your source directory by executing in the OpenFOAM-dev directory

git pull

The updates can then be compiled into OpenFOAM with the -update option (with or without the parallel -j option):

./Allwmake -update

If the update build does not compile correctly, e.g. after some very major changes, then the platform should be cleaned before recompiling with

wcleanPlatform

./Allwmake

# Ubuntu 上安装OpenFOAM

参考： https://zhuanlan.zhihu.com/p/601418292

sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key | apt-key add -"

sudo add-apt-repository http://dl.cfdem.cn/ubuntu

sudo apt-get update

sudo apt-get -y install openfoam10

echo "source /opt/openfoam10/etc/bashrc" >> ~/.bashrc

source ~/.bashrc

之后运行pisoFoam -help, 查看屏幕输出信息.

# 第一个算例 (方腔流)

https://doc.cfd.direct/openfoam/user-guide-v10/cavity

将示例中的cavity算例复制到$FOAM_RUN目录中.

mkdir -p $FOAM_RUN

cp -r $FOAM_TUTORIALS/incompressible/icoFoam/cavity/cavity $FOAM_RUN

cd $FOAM_RUN/cavity/

在$FOAM_RUN/cavity/目录依次运行如下命令:

blockMesh

icoFoam

paraFoam

终端结果如下, 这是计算结果.

并且会打开一个窗口，如下图:

点击Apply按钮，结果如图。这是后处理产生的图像。
