#!/bin/bash

# Simple script for running PFLOTRAN-OGS with less typing 

# Call this as:
# pft [infile] [number of processors]
# For example:
# pft spe10.in 4


# ****** IMPORTANT! **************************************************

# Please update this with the full path of your pflotran installation:
PFT_PATH=""
# For example:
#PFT_PATH="/home/myusername/pflotran"

# Please ensure the following variables are defined:
# PETSC_DIR
# PETSC_ARCH
# see "Installing and Running on Ubuntu" page for more information
#
# Don't forget to make this executable with chmod +x pft.sh

# ********************************************************************

$PETSC_DIR/$PETSC_ARCH/bin/mpiexec -n $2 $PFT_PATH/src/pflotran/pflotran -pflotranin $1
