# install adios2

## quick start
```
git clone https://github.com/ornladios/ADIOS2.git ADIOS2
mkdir adios2-build && cd adios2-build
cmake ../ADIOS2
make -j 16
make install
```

## CMake Options
```
      VAR				VALUE							Description
ADIOS2_USE_MPI			ON/OFF					MPI or non-MPI (serial) build.
ADIOS2_USE_ZeroMQ		ON/OFF					ZeroMQ for the DataMan engine.

ADIOS2_USE_HDF5			ON/OFF		HDF5 engine. If HDF5 is not on the syspath, it can be set using -DHDF5_ROOT=/path/to/hdf5

ADIOS2_USE_Python		ON/OFF		Python bindings. Python 3 will be used if found. If you want to specify a particular python version use -DPYTHON_EXECUTABLE=/path/to/interpreter/python

ADIOS2_USE_Fortran		ON/OFF		Bindings for Fortran 90 or above.

ADIOS2_USE_SST			ON/OFF		Simplified Staging Engine (SST) and its dependencies, requires MPI. Can optionally use LibFabric/UCX for RDMA transport. You can specify the LibFabric/UCX path manually with the -DLIBFABRIC_ROOT=¡­ or -DUCX_ROOT=¡­ option.

ADIOS2_USE_BZip2		ON/OFF		BZIP2 compression.
ADIOS2_USE_ZFP			ON/OFF		ZFP compression (experimental).
ADIOS2_USE_SZ			ON/OFF		SZ compression (experimental).
ADIOS2_USE_MGARD		ON/OFF		MGARD compression (experimental).
ADIOS2_USE_PNG			ON/OFF		PNG compression (experimental).
ADIOS2_USE_Blosc		ON/OFF		Blosc compression (experimental).
ADIOS2_USE_Endian_Reverse	ON/OFF	Enable endian conversion if a different endianness is detected between write and read.
ADIOS2_USE_IME			ON/OFF		DDN IME transport.
```

cmake -DADIOS2_USE_Fortran=ON -DADIOS2_USE_Python=OFF -DADIOS2_USE_ZeroMQ=OFF ../ADIOS2
 
## Installing the ADIOS2 library and the C++ and C bindings

By default, ADIOS2 will build the C++11 libadios2 library and the C and C++ bindings.

## Enabling the Python bindings

To enable the Python bindings in ADIOS2, based on PyBind11, make sure to follow these guidelines.

## Enabling the Fortran bindings

```
A Fortran 90 compliant compiler
A Fortran MPI implementation
```

## Running Tests
```
$ ctest
  or
$ make test
```

## Running Examples

https://github.com/ornladios/ADIOS2/tree/master/examples

```
A few very basic examples are described below:

Directory										Description
ADIOS2/examples/hello			very basic ¡°hello world¡±-style examples for reading and writing .bp files.
ADIOS2/examples/heatTransfer	2D Poisson solver for transients in Fourier¡¯s model of heat transfer. Outputs bp.dir or HDF5.
ADIOS2/examples/basics			covers different Variable use cases classified by the dimension.
```

## Linking ADIOS 2

### From CMake

ADIOS exports a CMake package configuration file that allows its targets to be directly imported into another CMake project via the find_package command:

```
cmake_minimum_required(VERSION 3.12)
project(MySimulation C CXX)

find_package(MPI REQUIRED)
find_package(ADIOS2 REQUIRED)
add_library(my_library src1.cxx src2.cxx)
target_link_libraries(my_library PRIVATE adios2::cxx11_mpi MPI::MPI_C)
```

### From non-CMake build systems

If you¡¯re not using CMake then you can manually get the necessary compile and link flags for your project using adios2-config:

```
$ /path/to/install-prefix/bin/adios2-config --cxxflags
ADIOS2_DIR: /path/to/install-prefix
-isystem /path/to/install-prefix/include -isystem /opt/ohpc/pub/mpi/openmpi3-gnu7/3.1.0/include -pthread -std=gnu++11
$ /path/to/install-prefix/bin/adios2-config --cxxlibs
ADIOS2_DIR: /path/to/install-prefix
-Wl,-rpath,/path/to/install-prefix/lib:/opt/ohpc/pub/mpi/openmpi3-gnu7/3.1.0/lib /path/to/install-prefix/lib/libadios2.so.2.4.0 -pthread -Wl,-rpath -Wl,/opt/ohpc/pub/mpi/openmpi3-gnu7/3.1.0/lib -Wl,--enable-new-dtags -pthread /opt/ohpc/pub/mpi/openmpi3-gnu7/3.1.0/lib/libmpi.so -Wl,-rpath-link,/path/to/install-prefix/lib
```
