# Compile the FORTRAN PETSc with MPICH2
#./config/configure.py --prefix=/home/petsc3.1 --with-cc='win32fe cl' --with-fc='win32fe ifort' --download-f-blas-lapack=1 --with-mpi-dir='/cygdrive/c/Cygwin/home/MPICH2' --with-debugging=0 

# For compiling C language PETSc library with MPICH2 for using VSL3D model.
# C Language Version
./config/configure.py --prefix=/home/petsc3.1-x64 --with-cc='win32fe cl' --with-fc=0 --with-cxx=0 --download-c-blas-lapack=1 --with-mpi-dir='/cygdrive/c/Cygwin/home/MPICH2_win64' --with-shared=0 --with-debugging=0 
# FORTRAN Language Version
#./config/configure.py --prefix=/home/petsc3.1-install --with-cc='win32fe cl' --with-fc='win32fe ifort' --with-cxx=0 --download-f-blas-lapack=1 --with-mpi-dir='/cygdrive/c/Cygwin/home/MPICH2' --with-shared=0 --with-debugging=0 


# HYPRE_v2.9 should be used.
# I tried:  But it did not work!
# --with-hypre-dir='/cygdrive/c/Cygwin/home/hypre-2.9.0b'

# The following format did not work because there's whitespace in the path name (Program Files (x86)).
#--with-mpi-dir='/cygdrive/c/Program Files (x86)/MPICH2' --with-mpiexec='mpiexec --localonly' 

# Another methods to install the external packages, I should try it:
# [1] Use --download-PACKAGENAME
# --download-mpich=/home/petsc/mpich2-1.0.4p1.tar.gz

# [2] Use --with-PACKAGENAME-dir=
# --with-PACKAGENAME-dir=PATH
# --with-hypre-dir=?

# [3] Use --with-PACKAGENAME-include   --with-PACKAGENAME-lib 
# --with-hypre-include=/home/
# --with-hypre-lib=/home/
# --with-metis-include=/home/metis-4.0.3/include --with-metis-lib=/home/metis-4.0.3/lib_win32/metis4.0.lib 
# --with-parmetis-include=/home/petsc/software/parmetis/include --with-parmetis-lib=/home/petsc/software/parmetis/lib/libparmetis.a