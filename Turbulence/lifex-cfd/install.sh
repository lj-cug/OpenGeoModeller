# Step 0 - Install lifex dependencies, 4 ways:
## (0) mk

## (1)lifex-env
git clone https://gitlab.com/lifex/lifex-env.git
cd lifex-env
./lifex-env.sh [options]
./lifex-env.sh --help
### The installation of any package can be enabled/disabled in the configuration file lifex-env.cfg.
## Run command:
./lifex-env.sh --prefix=/home/lijian/lifx-CFD/lifex-install -j 8 -y


### Add the following line to your ${HOME}/.bashrc file (or equivalent), replacing /path/to/lifex-env/ with the prefix where lifex-env has been installed:
source /home/lijian/lifx-CFD/lifex-env/configuration/enable_lifex.sh

## (2) Spack
### lifex dependencies are also available on Spack, a package manager for Linux and macOS.
### Add the following lines to your ${HOME}/.bashrc file (or equivalent), replacing /path/to/spack/ with the path where you want to install Spack:
export SPACK_ROOT=/home/lijian/lifx-CFD/
export PATH=${SPACK_ROOT}/bin:${PATH}

### Clone Spack with
mkdir -p ${SPACK_ROOT}
cd ${SPACK_ROOT}
git clone https://github.com/spack/spack.git ./

### Add the following line to your ${HOME}/.bashrc file (or equivalent):
source ${SPACK_ROOT}/share/spack/setup-env.sh

### Configure Spack as an environment module manager with
spack install environment-modules

### Add the following line to your ${HOME}/.bashrc file (or equivalent):
source $(spack location -i environment-modules)/init/bash

### Install deal.II and VTK (this step may require a long time to run) with
spack install dealii@9.3.1
spack install vtk@9.0.3 ^mesa~llvm

### Add the following lines to your ${HOME}/.bashrc file (or equivalent)
spack load openmpi
spack load cmake
spack load boost
spack load dealii
spack load vtk

### (3) Docker


# Step 1 - Install lifex
## Step 1.1 - Get lifex
cd /home/lijian/lifx-CFD/
git clone --recursive https://gitlab.com/lifex/lifx.git

## Step 1.2 - Configure lifex
cd /home/lijian/lifx-CFD/lifex-cfd
mkdir build
cd build
cmake -DLIN_ALG=Trilinos -DDEAL_II_DIR=/home/lijian/lifx-CFD/dealii/ -DVTK_DIR=/path/to/vtk/ \
	  -DBOOST_DIR=/path/to/boost/ -DMPIEXEC_PREFLAGS="--allow-run-as-root" ..
# -DCMAKE_BUILD_TYPE=Debug/Release
# Note (for developers): lifex can use either Trilinos or PETSc as linear algebra backends. The backend can be customized with -DLIN_ALG=Trilinos (the default value) or -DLIN_ALG=PETSc.
# Note (for advanced users): if you compiled and installed deal.II, VTK or Boost manually in any other way than mk, Spack or Docker, specify the installation directories with the cmake flags -DDEAL_II_DIR=/path/to/dealii/, -DVTK_DIR=/path/to/vtk/ and -DBOOST_DIR=/path/to/boost/.
# Note (for Docker users): please specify the cmake flag -DMPIEXEC_PREFLAGS="--allow-run-as-root" to enable the container to run parallel executables.

## Step 1.3 - Compile lifex
make -j<N>

#If you only need to compile part of the library, specific targets can be selected with, e.g.:
make -j<N> electrophysiology mechanics

## Step 1.4 - Check lifex installation
cd /path/to/lifex-cfd
git submodule update --init
make -j<N> setup_tests
ctest -L test_soft
# Note: by default, parallel tests related are run in parallel using MPI on all the processors detected on the host system. Configure lifex with the cmake flag -DMPIEXEC_MAX_NUMPROCS=<N> to set a custom number of parallel processes.

# Note (for developers): more computationally intensive tests can be run with ctest -L test_hard.

# You did it, have fun with lifex!

# Step 2 - What's next?
## reference:   https://lifex.gitlab.io/lifex-cfd/run.html

## Step 0 - Set the parameter file
./executable_name -g

## Step 1 - Run!
./executable_name -f custom_param_file.ext [option...]

### Parallel run
mpirun -n <N_PROCS> ./executable_name [option...]

### Dry run and parameter file conversion
./executable_name -l my_log_file.ext [option...]

### Wall time statistics
-t or --timer-statistics
