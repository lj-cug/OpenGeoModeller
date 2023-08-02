#!/bin/bash

#-------------- Environmental variables setup on Kunpeng Workstation -------------
# 加载编译环境
module use /workspace/public/software/modules
module load compilers/bisheng/2.1.0/bisheng2.1.0
module use /workspace/public/software/modules
module load mpi/hmpi/1.1.1/bisheng2.1.0
module use /workspace/public/software/modules
module load compilers/cuda/11.14.1


#---------------OP2 Migration on Kunpeng Workstation-----------------------------
export MY_PATH=/workspace/home/migration/zhusihai
export CUDA_INSTALL_PATH=/usr/local/cuda-11.4
export MPI_INSTALL_PATH=/workspace/public/software/mpi/hmpi/1.1.1/bisheng2.1.0/ompi
export PTSCOTCH_INSTALL_PATH=$MY_PATH/scotch_6.0.6
export PARMETIS_INSTALL_PATH=$MY_PATH/parmetis-4.0.3/metis
export HDF5_INSTALL_PATH=$MY_PATH/hdf5-hdf5-1_13_0/install

export OP2_COMPILER='clang'
export LD_LIBRARY_PATH=$CUDA_INSTALL_PATH/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PARMETIS_INSTALL_PATH/lib:$PTSCOTCH_INSTALL_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HDF5_INSTALL_PATH/lib:$LD_LIBRARY_PATH

# Prepare installpath of OP2 for building MG-CFD-OP2
export OP2_INSTALL_PATH=$MY_PATH/OP2-Common-release-2020/op2/c

export NV_ARCH='Ampere'

make -j$(nproc)

#-----------------------Ubuntu build parallel with MPI--------------------
export CUDA_INSTALL_PATH=/usr/local/cuda-11.0/targets/x86_64-linux
export PTSCOTCH_INSTALL_PATH=/home/lijian/DSL-OP2-Kunpeng/scotch_6.0.6
export PARMETIS_INSTALL_PATH=/home/lijian/DSL-OP2-Kunpeng/parmetis-4.0.3-install
export HDF5_INSTALL_PATH=/home/lijian/DSL-OP2-Kunpeng/hdf5-1.12.1-parallel-install
export MPI_INSTALL_PATH=/home/lijian/openmpi-3.1
export OP2_COMPILER='gnu'

export LD_LIBRARY_PATH=$CUDA_INSTALL_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PARMETIS_INSTALL_PATH/lib:$PTSCOTCH_INSTALL_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HDF5_INSTALL_PATH/lib:$LD_LIBRARY_PATH
export NV_ARCH='Pascal'


#-----------------------Ubuntu build serial--------------------
export CUDA_INSTALL_PATH=/usr/local/cuda-11.0/targets/x86_64-linux
export HDF5_INSTALL_PATH=/home/lijian/DSL-OP2-Kunpeng/hdf5-1.12.1-serial-install
export OP2_COMPILER='gnu'
export LD_LIBRARY_PATH=$CUDA_INSTALL_PATH/lib:$HDF5_INSTALL_PATH/lib:$LD_LIBRARY_PATH
export NV_ARCH='Pascal'

# build MG-CFD-OP2
export OP2_INSTALL_PATH=/home/lijian/DSL-OP2-Kunpeng/OP2-Common-release-2020/op2/c



