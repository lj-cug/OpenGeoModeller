cd $PROGS
wget http://zlib.net/zlib-1.2.8.tar.gz 
tar -zxvf zlib-1.2.8.tar.gz
cd zlib-1.2.8
export CC=gcc
export FC=gfortran
./configure --prefix=`pwd`
make
make install