# download
git clone -b release https://gitlab.com/petsc/petsc.git petsc
cd petsc
git pull
PETSc_VERSION=3.15.5    # 更改PETSc的版本
git checkout v${PETSc_VERSION}

#
# If you do not have a Fortran compiler or MPICH installed locally (and want to use PETSc from C only).
#./configure --prefix=/home/user/soft/petsc-install --with-cc=gcc --with-cxx=0 --with-fc=0 --download-f2cblaslapack --download-mpich

# 建议使用：
apt-get install valgrind

# If BLAS/LAPACK, MPI are already installed in known user location use:
./configure --prefix=/home/lijian/HPC_Build/petsc-${PETSc_VERSION}-install --with-blaslapack-dir=/usr/lib/x86_64-linux-gnu --with-cc=`which mpicc` --with-mpi-f90=`which mpif90` --with-mpiexec=`which mpiexec`

./configure --prefix=/home/lijian/HPC_Build/petsc-${PETSc_VERSION}-install --with-blaslapack-dir=/usr/lib/x86_64-linux-gnu --with-mpi-dir=/home/lijian/openmpi-3.1

# 用的这个:waiwer需要使用PETSc-HDF5
./configure --prefix=/home/lijian/HPC_Build/petsc-${PETSc_VERSION}-install --with-blaslapack-dir=/usr/lib/x86_64-linux-gnu --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 --with-hdf5-dir=/home/lijian/HPC_Build/hdf5-1.14.1-install --with-debugging=0


# Build Complex version of PETSc (using c++ compiler):
./configure --with-cc=gcc --with-fc=gfortran --with-cxx=g++ --with-clanguage=cxx --download-fblaslapack --download-mpich --with-scalar-type=complex


--download-openmpi
--download-fblaslapack
export PETSC_ARCH=arch-opt
--with-debugging=0



# Install 2 variants of PETSc, one with gnu, the other with Intel compilers.
./configure PETSC_ARCH=linux-gnu --with-cc=gcc --with-cxx=g++ --with-fc=gfortran --download-mpich
make PETSC_ARCH=linux-gnu all test
./configure PETSC_ARCH=linux-gnu-intel --with-cc=icc --with-cxx=icpc --with-fc=ifort --download-mpich --with-blaslapack-dir=/usr/local/mkl
make PETSC_ARCH=linux-gnu-intel all test



# External Packages; One can optionally use external solvers like HYPRE, MUMPS, and others from within PETSc applications.
./configure --download-fblaslapack --download-mpich
./configure --with-superlu-include=/home/petsc/software/superlu/include --with-superlu-lib=/home/petsc/software/superlu/lib/libsuperlu.a
./configure --with-parmetis-include=/sandbox/balay/parmetis/include --with-parmetis-lib="-L/sandbox/balay/parmetis/lib -lparmetis -lmetis"
./configure --with-parmetis-include=/sandbox/balay/parmetis/include --with-parmetis-lib=[/sandbox/balay/parmetis/lib/libparmetis.a,libmetis.a]


# cuda
#In most cases you need only pass the configure option --with-cuda


# build and install
export PETSC_DIR=/home/lijian/HPC_Build/waiwera/petsc
export PETSC_ARCH=arch-linux-c-debug

make all check
make install


# 可以参考  https://www.underworldcode.org/articles/setting-up-underworld-dependencies/
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
    
    
