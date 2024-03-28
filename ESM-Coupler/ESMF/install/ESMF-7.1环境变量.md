# ESMF-7.1的环境变量

```
export ESMF_OS=Linux
export ESMF_ABI=64
export ESMF_DIR=/home/lijian/esmf-7.1.0
export ESMF_INSTALL_PREFIX=/home/lijian/esmf-7.1.0/install_dir

export ESMF_COMM=openmpi
export ESMF_BOPT=o
export ESMF_OPENMP=OFF

export ESMF_COMPILER=gfortran
export ESMF_CXXCOMPILER=mpic++
export ESMF_F90COMPILER=mpif90

export ESMF_NETCDF=split
export ESMF_NETCDF_INCLUDE="/usr/include"
export ESMF_NETCDF_LIBPATH="/usr/lib/x86_64-linux-gnu"
export ESMF_NETCDF_LIBS="-lnetcdf -lnetcdff"
export ESMF_PIO=internal
export ESMF_XERCES=standard
export ESMF_XERCES_INCLUDE="/usr/include"
export ESMF_XERCES_LIBPATH="/usr/lib/x86_64-linux-gnu"

export ESMF_SITE=default
export ESMF_TESTMPMD=OFF
export ESMF_TESTEXHAUSTIVE=OFF
export ESMF_TESTHARNESS_ARRAY=RUN_ESMF_TestHarnessArray_default
export ESMF_TESTHARNESS_FIELD=RUN_ESMF_TestHarnessField_default
export ESMF_TESTWITHTHREADS=OFF
export ESMF_TESTEXHAUSTIVE=ON
export ESMF_YAMLCPP=OFF
```

## Add the path of libraries as environmental variables

```
export ESMF_INC=$ESMF_INSTALL_PREFIX/include
export ESMF_LIB=$ESMF_INSTALL_PREFIX/lib/libO/Linux.gfortran.64.openmpi.default
export ESMFMKFILE=$ESMF_LIB/esmf.mk
```

## Build

```
make -j8
make build_system_tests
make check 
make install
```