# install ParaView-v5.9.1

## ����Դ��

git clone https://gitlab.kitware.com/paraview/paraview.git

���ߣ�

git clone -b v5.9.1 --recursive https://gitlab.kitware.com/paraview/paraview.git paraview-5.9.1

cd paraview

git submodule update --init --recursive

## ���´���

git pull

git submodule update

## ��������CMAKE 

cd ../

mkdir build

cd build

cmake .. 
 
## ��װ

make -j8

make install