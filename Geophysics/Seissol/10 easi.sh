# easi is a library written in C++14. It needs to be compiled with Cmake.
# Easi depends on the following three projects:
#yaml-cpp
#ASAGI
#ImpalaJIT
#Lua

export FC=mpif90
export CXX=mpiCC
export CC=mpicc

# build yaml-cpp
git clone git@github.com:jbeder/yaml-cpp
# git clone https://github.com/jbeder/yaml-cpp.git
cd yaml-cpp
git checkout yaml-cpp-0.6.3
mkdir build && cd build
cmake ..  -DCMAKE_INSTALL_PREFIX=$HOME
make -j 4
make install
cd ../..

# build ImpalaJIT
git clone https://github.com/uphoffc/ImpalaJIT.git
cd ImpalaJIT
mkdir build && cd build
cmake ..  -DCMAKE_INSTALL_PREFIX=$HOME
make -j 4
make install
cd ../..

# Lua
wget https://www.lua.org/ftp/lua-5.3.6.tar.gz
tar -xzvf lua-5.3.6.tar.gz
cd lua-5.3.6/
make linux CC=mpicc
make local
cp -r install/* ~
cd ..

# build easi
git clone https://github.com/SeisSol/easi.git

# $EASI_SRC is the directory into which you've cloned the easi source
cmake -DCMAKE_PREFIX_PATH=$HOME -DCMAKE_INSTALL_PREFIX=$HOME -DASAGI=ON -DIMPALAJIT=ON -DLUA=ON $EASI_SRC
make -j4 install

