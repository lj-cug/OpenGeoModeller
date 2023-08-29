# Obtain GCC source
curl -L -O http://mirrors.concertpass.com/gcc/releases/gcc-10.4.0/gcc-10.4.0.tar.gz
tar -xf gcc-10.4.0.tar.gz -C .

# Obtain GCC pre-reqs
cd gcc-10.4.0
./contrib/download_prerequisites

# Configure GCC using the recommended arguments
mkdir gcc-build
cd gcc-build

../configure --prefix=/target/installation/path/gcc-10.4.0 \
--disable-multilib \
--enable-languages=c,c++,fortran,jit \
--enable-checking=release \
--enable-host-shared \
--with-pic

# With configure complete (and error free), build and install GCC
make -j 6
make install

