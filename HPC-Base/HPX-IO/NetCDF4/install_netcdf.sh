# build netcdf-4.8.1
NCDIR=/home/lijian/HPX/netcdf4.8-install
installpath=/home/lijian/DSL-OP2-Kunpeng/hdf5-1.12.1-serial-install
CC=/usr/bin/gcc CXX=/usr/bin/g++ CPPFLAGS="-I$installpath/include" \
CFLAGS="-I$installpath/include" \
LDFLAGS="-L$installpath/lib" \
./configure --prefix=${NCDIR}  --enable-shared --enable-netcdf-4 --disable-doxygen --enable-static --enable-largefile
make
make install
export LD_LIBRARY_PATH=${NCDIR}/lib:${LD_LIBRARY_PATH}


CC=/usr/bin/gcc
FC=/usr/bin/gfortran
export LD_LIBRARY_PATH=${NFDIR}/lib:${LD_LIBRARY_PATH}
NFDIR=/home/lijian/HPX/netcdff4.5-install
CPPFLAGS=-I${NCDIR}/include LDFLAGS=-L${NCDIR}/lib \
./configure --prefix=${NFDIR}
make -j$(nproc)
make install


