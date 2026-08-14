# Install Firedrake
https://www.firedrakeproject.org/install.html#install-system-dependencies

https://www.firedrakeproject.org/install.html#installing-firedrake-using-pip

A native installation of Firedrake is accomplished in 3 steps:
1. Install system dependencies
2. Install PETSc
3. Install Firedrake

## Prerequisites
On Linux the only prerequisite needed to install Firedrake is a suitable version of Python (3.10 or greater).

## firedrake-configure
To simplify the installation process, Firedrake provides a utility script called firedrake-configure. 
This script can be downloaded by executing:
```
$ curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/release/scripts/firedrake-configure
```

Note that firedrake-configure does not install Firedrake for you. It is simply a helper script that emits the configuration options that Firedrake needs for the various steps needed during installation.

This means that if you want to install Firedrake in a non-standard way (for instance with a custom installation of PETSc, HDF5 or MPI) then it is your responsibility to modify the output from firedrake-configure as necessary. This is described in more detail in ** Customising Firedrake **.

## Installing system dependencies
If on Ubuntu (24.04 - 26.04) or macOS, system dependencies can be installed with firedrake-configure. On Ubuntu run:
```
sudo apt install $(python3 firedrake-configure --show-system-packages)
```
which will install the following packages:
```
bison build-essential cmake flex gfortran git ninja-build pkg-config python3-dev python3-pip libfftw3-dev libfftw3-mpi-dev libhwloc-dev libhdf5-mpi-dev libmumps-ptscotch-dev libmetis-dev libnetcdf-dev libopenblas-dev libopenmpi-dev libpnetcdf-dev libptscotch-dev libscalapack-openmpi-dev libsuitesparse-dev libsuperlu-dev libsuperlu-dist-dev
```
The packages installed here are a combination of system dependencies, like a C compiler, BLAS, and MPI, and ¡®external packages¡¯ that are used by PETSc, like MUMPS and HDF5.

If you are not installing onto Ubuntu or macOS then it is your responsibility to ensure that these system dependencies are in place. Some of the dependencies (e.g. a C compiler) must come from your system whereas others, if desired, may be downloaded by PETSc configure by passing additional flags like --download-mpich or --download-openblas (run ./configure --help | less to see what is available). To give you a guide as to what system dependencies are needed, on Ubuntu they are:
```
build-essential flex gfortran git ninja-build pkg-config python3-dev python3-pip
```

## Installing PETSc





## Installing Firedrake




### Updating Firedrake



## Common installation issues
