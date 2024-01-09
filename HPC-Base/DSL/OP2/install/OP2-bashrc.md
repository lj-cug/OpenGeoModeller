# OP2-环境变量设置
```
# options for compiling OP2 library
export OP2_COMPILER=gnu
export CUDA_INSTALL_PATH=/usr/local/cuda
export NV_ARCH=Pascal    # Geforce MX150

export PTSCOTCH_INSTALL_PATH=/home/lijian/DSL/ptscotch
export PARMETIS_INSTALL_PATH=/home/lijian/DSL/parmetis-4.0
export HDF5_PAR_INSTALL_PATH=/usr/lib/x86_64-linux-gnu/hdf5/mpich

export LD_LIBRARY_PATH=/home/lijian/DSL/parmetis-4.0/lib:/home/lijian/DSL/ptscotch/lib:/$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/hdf5/mpich/lib:$LD_LIBRARY_PATH

export OP2_INSTALL_PATH=~/DSL/OP2-Common/op2/

#export OP2_C_COMPILER=gcc
#export OP2_C_CUDA_COMPILER=nvcc
#export OP2_F_COMPILER=gfortran
#----------------------------------------------
# options for compiling OP2 library
export OP2_COMPILER=gnu
#export OP2_C_COMPILER=gnu
#export OP2_C_CUDA_COMPILER=nvhpc
#export OP2_F_COMPILER=gnu

export OP2_ROOT=/home/lijian/DSL/OP2-Common-release-2020
export OP2_INSTALL_PATH=$OP2_ROOT/op2
unset OP_AUTO_SOA
export PYTHONPATH=$OP2_ROOT/translator/c/python:$PYTHONPATH

#External libraries
export PTSCOTCH_INSTALL_PATH=/home/lijian/DSL/ptscotch
export PARMETIS_INSTALL_PATH=/home/lijian/DSL/parmetis-4.0
export LD_LIBRARY_PATH=$PARMETIS_INSTALL_PATH/lib:$PTSCOTCH_INSTALL_PATH/lib:$LD_LIBRARY_PATH
export HDF5_PAR_INSTALL_PATH=/usr/lib/x86_64-linux-gnu/hdf5/mpich
export LD_LIBRARY_PATH=$HDF5_PAR_INSTALL_PATH/lib:$LD_LIBRARY_PATH

#NVIDIA CUDA
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_INSTALL_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export NV_ARCH=Pascal    # Geforce MX150


# include path
export C_INCLUDE_PATH=$CUDA_INSTALL_PATH/include:$MPI_INSTALL_PATH/include:$HDF5_PAR_INSTALL_PATH/include:$PTSCOTCH_INSTALL_PATH/include:$CPLUS_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDA_INSTALL_PATH/include:$MPI_INSTALL_PATH/include:$HDF5_PAR_INSTALL_PATH/include:$PTSCOTCH_INSTALL_PATH/include:$CPLUS_INCLUDE_PATH

# library path
#export HAVE_CUDA=true
#export CUDA_LIB=/usr/local/cuda/lib64
#export HAVE_PTSCOTCH=true
#export PTSCOTCH_LIB=$PTSCOTCH_INSTALL_PATH/lib
#export HAVE_PARMETIS=true
#export PARMETIS_LIB=$PARMETIS_INSTALL_PATH/lib
#export HAVE_HDF5_PAR=true
#export HDF5_PAR_LIB=$HDF5_PAR_INSTALL_PATH/lib
```