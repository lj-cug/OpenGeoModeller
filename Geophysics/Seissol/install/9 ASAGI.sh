# The software package ASAGI can be used to map gridded simulation properties of the domain to the mesh used for a SeisSol simulation. ASAGI reads NetCDF files, which follow the COARDS Convention for netCDF files. 

# https://seissol.readthedocs.io/en/latest/asagi.html#installing-asagi

# Be careful that the python and gcc package is the same as for the compilation of SeisSol in a later step! 

git clone https://github.com/TUM-I5/ASAGI.git
cd ASAGI
git submodule update --init

export FC=mpif90
export CXX=mpiCC
export CC=mpicc

mkdir build && cd build
CMAKE_PREFIX_PATH=$NETCDF_BASE
cmake .. -DSHARED_LIB=no -DSTATIC_LIB=yes -DCMAKE_INSTALL_PREFIX=$HOME
make -j 48
make install
#(Know errors: 1.Numa could not found - turn off Numa by adding -DNONUMA=on . )

# building SeisSol with ASAGI support
# Simply turn on the option ASAGI=ON in the using ccmake.