# ESMF-8.0环境变量

使用MPICH2编译的

```
#------------config for ESMF-8.0------------------------

## Set the path for MPI and NETCDF
export NETCDF_DIR="/usr/local/"
export PNETCDF_DIR="/usr/local/"
export MPI_DIR="/home/lijian/mpich-3.3"

## Set ESMF code path
export ESMF_DIR="/home/lijian/esmf-8.0.1"
export ESMF_LIB=$ESMF_DIR/lib/libO3/Linux.gfortran.64.mpich2.default/
export ESMF_INSTALL_PREFIX="/usr/local/esmf/"
export ESMFMKFILE=$ESMF_LIB/esmf.mk

############################
export ESMF_OS=Linux
export ESMF_COMM=openmpi
export ESMF_OPENMP=OFF
export ESMF_LAPACK=internal
export ESMF_BOPT=O
export ESMF_ABI=64
export ESMF_COMPILER=gfortran
export ESMF_CXXCOMPILER=mpicxx
export ESMF_F90COMPILER=mpifort

export ESMF_NETCDF="standard"
export ESMF_NETCDF_INCLUDE="/usr/local/include/"
export ESMF_NETCDF_LIBPATH="/usr/local/lib/"
export ESMF_NETCDF_LIBS="-lnetcdf -lnetcdff"
export ESMF_PNETCDF="standard"
export ESMF_PNETCDF_INCLUDE="/usr/local/include/"
export ESMF_PNETCDF_LIBPATH="/usr/local/lib/"
export ESMF_PNETCDF_LIBS="-lpnetcdf"
export ESMF_PIO=OFF

export ESMF_YAMLCPP=OFF
export ESMF_TESTEXHAUSTIVE=ON
export ESMF_TESTMPMD=OFF
export ESMF_TESTSHAREDOBJ=ON
export ESMF_TESTWITHTHREADS=OFF

### Add the path of libraries as environmental variables
export LD_LIBRARY_PATH=$NETCDF_DIR/lib/${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$PNETCDF_DIR/lib/${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$MPI_DIR/lib/${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$MPI_DIR/bin/${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$ESMF_LIB/${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

#---------------end of ESMF-8.0 configuration----------------------------
```








