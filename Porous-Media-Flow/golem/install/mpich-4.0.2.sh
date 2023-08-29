# Download MPICH 4.0.2
curl -L -O http://www.mpich.org/static/downloads/4.0.2/mpich-4.0.2.tar.gz
tar -xf mpich-4.0.2.tar.gz -C .

# Create an out-of-tree build location and configure MPICH using the recommended arguments
mkdir mpich-4.0.2/gcc-build
cd mpich-4.0.2/gcc-build

../configure --prefix=/target/installation/path/mpich-4.0.2 \
--enable-shared \
--enable-sharedlibs=gcc \
--enable-fast=O2 \
--enable-debuginfo \
--enable-totalview \
--enable-two-level-namespace \
CC=gcc \
CXX=g++ \
FC=gfortran \
F77=gfortran \
F90='' \
CFLAGS='' \
CXXFLAGS='' \
FFLAGS='-fallow-argument-mismatch' \
FCFLAGS='-fallow-argument-mismatch' \
F90FLAGS='' \
F77FLAGS=''

# With configure complete and error free, build and install MPICH
make -j 6
make install

