#!/bin/bash

# DAFoam root path
export DAFOAM_ROOT_PATH=$HOME/dafoam
# Miniconda3
export PATH=$DAFOAM_ROOT_PATH/packages/miniconda3/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DAFOAM_ROOT_PATH/packages/miniconda3/lib
export PYTHONUSERBASE=no-local-libs
# OpenMPI-3.1.6
export MPI_INSTALL_DIR=$DAFOAM_ROOT_PATH/packages/openmpi-3.1.6/opt-gfortran
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPI_INSTALL_DIR/lib
export PATH=$MPI_INSTALL_DIR/bin:$PATH
# PETSC
export PETSC_DIR=$DAFOAM_ROOT_PATH/packages/petsc-3.14.6
export PETSC_ARCH=real-opt
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PETSC_DIR/$PETSC_ARCH/lib
export PETSC_LIB=$PETSC_DIR/$PETSC_ARCH/lib
# CGNS-4.1.2
export CGNS_HOME=$DAFOAM_ROOT_PATH/packages/CGNS-4.1.2/opt-gfortran
export PATH=$PATH:$CGNS_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CGNS_HOME/lib
# Ipopt
export IPOPT_DIR=$DAFOAM_ROOT_PATH/packages/Ipopt
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$IPOPT_DIR/lib
# OpenFOAM-v1812
source $DAFOAM_ROOT_PATH/OpenFOAM/OpenFOAM-v1812/etc/bashrc
export LD_LIBRARY_PATH=$DAFOAM_ROOT_PATH/OpenFOAM/sharedLibs:$LD_LIBRARY_PATH
export PATH=$DAFOAM_ROOT_PATH/OpenFOAM/sharedBins:$PATH
