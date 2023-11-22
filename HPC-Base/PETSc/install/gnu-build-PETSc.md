# Download PETSc_{VERSION}

```
git clone -b release https://gitlab.com/petsc/petsc.git petsc
cd petsc
git pull
PETSc_VERSION=3.15.5
git checkout v${PETSc_VERSION}
```

# 配置编译

使用优化版本的PETSc, ./configure时都要加上：--with-debugging=0

## 仅使用C语言, 且没有安装MPI库

./configure --prefix=$PWD/install --with-cc=gcc --with-cxx=0 --with-fc=0 --with-debugging=0 --download-f2cblaslapack --download-mpich

这里,如果需要使用openmpi库, 则使用： --download-openmpi

如果使用Fortran版本的BLAS库，则使用： --download-fblaslapack

## 已经安装MPI和BLAS库

```
./configure --prefix=$PWD/petsc-${PETSc_VERSION}-install --with-blaslapack-dir=/usr/lib/x86_64-linux-gnu --with-cc=`which mpicc` --with-mpi-f90=`which mpif90` --with-mpiexec=`which mpiexec`

./configure --prefix=/home/lijian/HPC_Build/petsc-${PETSc_VERSION}-install --with-blaslapack-dir=/usr/lib/x86_64-linux-gnu --with-mpi-dir=/home/lijian/openmpi-3.1
```

## 使用HDF5第3方库的PETSc

waiwera应用需要HDF5格式输出

./configure --prefix=$PWD/petsc-${PETSc_VERSION}-install --with-blaslapack-dir=/usr/lib/x86_64-linux-gnu --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 --with-hdf5-dir=/home/lijian/HPC_Build/hdf5-1.14.1-install --with-debugging=0

## 使用Intel编译器

./configure PETSC_ARCH=linux-gnu-intel --with-cc=icc --with-cxx=icpc --with-fc=ifort --download-mpich --with-blaslapack-dir=/usr/local/mkl

## 使用一些额外的软件包, 如HYPRE, MUMPS等

```
./configure --download-fblaslapack --download-mpich
./configure --with-superlu-include=/home/petsc/software/superlu/include --with-superlu-lib=/home/petsc/software/superlu/lib/libsuperlu.a
./configure --with-parmetis-include=/sandbox/balay/parmetis/include --with-parmetis-lib="-L/sandbox/balay/parmetis/lib -lparmetis -lmetis"
./configure --with-parmetis-include=/sandbox/balay/parmetis/include --with-parmetis-lib=[/sandbox/balay/parmetis/lib/libparmetis.a,libmetis.a]
```

# 完成配置后编译
```
make all check
make install
```

编译完成后，设置好2个环境变量：
```
export PETSC_DIR=/path/to/code-of-PETSc
export PETSC_ARCH=arch-linux-c-opt 
```

# Installing PETSc to use NVidia GPUs (aka CUDA)

   Install CUDA, Thrust, Cusp in default locations; /usr/local/cuda. 
   The versions and locations of Thrust and Cusp you should use are listed in $PETSC_DIR/config/PETSc/packages/cuda.py. 
   The required version of CUDA may not be publically available, you may need to register as an NVIDIA Developer (free) to access them. 
   
Make sure nvcc is in PATH

设置好CUDA lib64的路径：

install compatible NVidia kernel developer driver by running the executable you download, as root
make sure  LD_LIBRARY_PATH is set to point to the CUDA libraries, for instance
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib


Configure and build PETSc with the additional configure options
--with-cuda=1 --with-cusp=1 --with-thrust=1

if the GPU card only supports single precision add --with-precision=single
if you did not install in default locations add --with-thrust-dir=path_to_thrust and --with-cusp-dir=path_to_cusp

# 运行测试例子

cd src/snes/examples/tutorials

make ex19

./ex19 -da_vec_type mpicusp -da_mat_type mpiaijcusp -pc_type none -dmmg_nlevels 1 -da_grid_x 100 -da_grid_y 100 -log_summary -mat_no_inode -preload off  -cusp_synchronize

We only have experience using Nvidia GPUs on Apple machines and Intel Xeon servers running Ubuntu 10.04 x86_64 NVidia GT200 [Tesla C1060] 

# 更多的编译参数配置

可以参考  https://www.underworldcode.org/articles/setting-up-underworld-dependencies/

./configure \
    --prefix=/opt/petsc/${PETSC_VERSION} \
    --with-debugging=yes                 \
    --COPTFLAGS="-O3"                 \
    --CXXOPTFLAGS="-O3"               \
    --FOPTFLAGS="-O3"                 \
    --with-shared-libraries              \
    --with-cxx-dialect=C++11             \
    --with-make-np=8                     \
    --download-mpich=yes                 \
    --download-hdf5=yes                  \
    --download-mumps=yes                 \
    --download-parmetis=yes              \
    --download-metis=yes                 \
    --download-superlu=yes               \
    --download-hypre=yes                 \
    --download-superlu_dist=yes          \
    --download-scalapack=yes             \
    --download-cmake=yes