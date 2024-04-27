# install ParaView-v5.9.1

## 下载源码

git clone https://gitlab.kitware.com/paraview/paraview.git

或者：

git clone -b v5.9.1 --recursive https://gitlab.kitware.com/paraview/paraview.git paraview-5.9.1

cd paraview

git submodule update --init --recursive

## 更新代码

git pull

git submodule update

## 初步设置CMAKE 

cd ../

mkdir build

cd build

cmake .. 
 
## 安装

make -j8

make install