# main program
export PATH=$PATH:/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin

# CUDA 9.0 library
export PATH=/usr/local/cuda-9.0/include:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH

# PGI compiler
export PGI=/opt/pgi
export PATH=/opt/pgi/linux86-64/19.4/bin:$PATH
export MANPATH=$MANPATH:/opt/pgi/license.dat
export PGI_ACC_TIME=1
export PGI_ACC_NOTIFY=1

# intel compiler
export PATH=/opt/intel/bin/:$PATH

##PALM LES atmosphere model configuration##
export PALM_BIN=/root/palm/current_version/trunk/SCRIPTS
export PATH=$PALM_BIN:$PATH

# Tecplot 2018
export PATH="/usr/local/tecplot/360ex_2018r1/bin:$PATH"

# set MPICH2-3.3 or openmpi-3.1
export PATH=/home/lijian/mpich-3.3/bin:$PATH

# CMAKE-3.17 or CMAKE-3.8
export PATH=/home/lijian/Downloads/cmake-3.17.5/bin:$PATH
#export PATH=/home/lijian/Downloads/cmake-3.8.0/bin:$PATH

# KeckCAVE
export PATH=$PATH:/home/lijian/KeckCAVE/3DVisualizer-1.15/bin
export PATH=$PATH:/home/lijian/KeckCAVE/LiDARViewer-2.13/bin

# AMGX-wrapper and PETSc library for shyFEM
export PETSC_DIR=//home/lijian/AMG/petsc-3.8.4/arch-linux2-c-opt
export LD_LIBRARY_PATH=/home/lijian/AMG/amgx-c-wrapper-master/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/lijian/AMG/AmgXWrapper-master/build/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/lijian/AMG/petsc-3.8.4/arch-linux2-c-opt/lib:$LD_LIBRARY_PATH

