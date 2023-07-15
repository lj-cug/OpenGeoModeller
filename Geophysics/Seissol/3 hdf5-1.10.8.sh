wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.8/src/hdf5-1.10.8.tar.bz2
tar -xaf hdf5-1.10.8.tar.bz2
cd hdf5-1.10.8
CPPFLAGS="-fPIC ${CPPFLAGS}" CC=mpicc FC=mpif90 ./configure --enable-parallel --prefix=$HOME --with-zlib --disable-shared --enable-fortran
make -j8
make install
cd ..