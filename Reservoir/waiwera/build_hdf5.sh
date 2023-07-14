wget -c https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_14_1.tar.gz
tar xvf hdf5-1_14_1.tar.gz
rm hdf5-1_14_1.tar.gz

cd hdf5-hdf5-1_14_1

./configure --prefix=/home/lijian/HPC_Build/hdf5-1.14.1-install --enable-fortran --enable-cxx


make -j4
make check
make install
make check-install 


