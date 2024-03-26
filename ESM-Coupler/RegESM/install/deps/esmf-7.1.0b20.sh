# Before starting to the installation of ESMF library, the users need to pay attention to the following issues;
#* The coupled model needs special features of ESMF (>7.1.0b20). 
#* If there is a plan to use the debug level (>2) to check the exchange fields, then ESMF library might be installed with parallel I/O and netCDF support (for more information, look at the example environment variables defined in ESMF installation related with NETCDF and PNETCDF).
#* After netCDF version 4.3.0 the C++ interface is changed and ESMF is not compatible with it. So, it is better to use the <= 4.3.0 version of netCDF in this case.

## Example environment variable definitions (sh/bash shell) for ESMF installations:
export ESMF_OS=Linux
export ESMF_TESTMPMD=OFF
export ESMF_TESTHARNESS_ARRAY=RUN_ESMF_TestHarnessArray_default
export ESMF_TESTHARNESS_FIELD=RUN_ESMF_TestHarnessField_default
export ESMF_DIR=/okyanus/users/uturuncoglu/progs/esmf-7.1.0b20
export ESMF_TESTWITHTHREADS=OFF
export ESMF_INSTALL_PREFIX=$PROGS/esmf-7.1.0b20/install_dir
export ESMF_COMM=intelmpi
export ESMF_TESTEXHAUSTIVE=ON
export ESMF_BOPT=O
export ESMF_OPENMP=OFF
export ESMF_SITE=default
export ESMF_ABI=64
export ESMF_COMPILER=intel
export ESMF_PIO=internal
export ESMF_NETCDF=split
export ESMF_NETCDF_INCLUDE=$PROGS/netcdf-4.4.0/include
export ESMF_NETCDF_LIBPATH=$PROGS/netcdf-4.4.0/lib
export ESMF_XERCES=standard
export ESMF_XERCES_INCLUDE=$PROGS/xerces-c-3.1.4/include
export ESMF_XERCES_LIBPATH=$PROGS/xerces-c-3.1.4/lib

## Then, following command can be used to compile and install ESMF
cd $ESMF_DIR 
make >&make.log 
make install

##After installation of the ESMF library, user can create a new environment variables like **ESMF_INC**, **ESMF_LIB** and **ESMFMKFILE** to help RegESM configure script to find the location of the required files for the ESMF library.
export ESMF_INC=$ESMF_INSTALL_PREFIX/include
export ESMF_LIB=$ESMF_INSTALL_PREFIX/lib/libO/Linux.intel.64.intelmpi.default
export ESMFMKFILE=$ESMF_INSTALL_PREFIX/lib/libO/Linux.intel.64.intelmpi.default/esmf.mk


