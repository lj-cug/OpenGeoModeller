wget https://syncandshare.lrz.de/dl/fiJNAokgbe2vNU66Ru17DAjT/netcdf-4.6.1.tar.gz
tar -xaf netcdf-4.6.1.tar.gz
cd netcdf-4.6.1
CFLAGS="-fPIC ${CFLAGS}" CC=h5pcc ./configure --enable-shared=no --prefix=$HOME --disable-dap
#NOTE: Check for this line to make sure netCDF is built with parallel I/O:
#"checking whether parallel I/O features are to be included... yes" This line comes at the very end (last 50 lines of configure run)!
make -j8
make install
cd ..