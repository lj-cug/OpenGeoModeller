# For compiling C and FORTRAN PETSc library with MPICH2, METIS and fblaslapck for using in defmod model.
./config/configure.py --prefix=/home/petsc-3.7-install --with-cc='win32fe cl' --with-fc='win32fe ifort' --download-fblaslapack --with-debugging=0 --with-shared=0 --with-mpi-dir='/cygdrive/c/Cygwin/home/MPICH2' 


# The format did not work!
#--with-metis-include='/cygdrive/c/Cygwin/home/metis-4.0.3/include' --with-metis-lib='/cygdrive/c/Cygwin/home/metis-4.0.3/lib/metis4.0.lib' 

# The following format did not work too!
# --with-metis-dir='/cygdrive/c/Cygwin/home/metis-4.0.3' 

# So, I have to compile petsc-metis using Cmake tool, and link the petsc and metis libs in VS when compiling defmod.
# Then, firstly I compiled the PETsc3.8 with MPICH2,
# secondly, I compiled METIS4.0 in Intel Visual Fortran compiler.

# In Cygwin OS
#./config/configure.py --prefix=/home/petsc3.8 --with-cc=mpicc --with-fc=mpif90 --with-cxx=mpicxx --download-fblaslapack  --download-metis --with-debugging=0
