# Install PFLOTRAN-OGS (Ubuntu 20.04)
## install PETSc
```
cd petsc
git checkout v3.19.1
./configure --download-mpich=yes --download-hdf5=yes --download-fblaslapack=yes --download-metis=yes  --download-cmake=yes --download-ptscotch=yes --download-hypre=yes --with-debugging=0 COPTFLAGS=-O3 CXXOPTFLAGS=-O3 FOPTFLAGS=-O3
make PETSC_DIR=/home/myusername/petsc PETSC_ARCH=ubuntu-opt all
make PETSC_DIR=/home/myusername/petsc PETSC_ARCH=ubuntu-opt check
```
## Installing PFLOTRAN-OGS_1.8
```
cd pflotran_ogs_1.8/src/pflotran
make -j4 pflotran
```

This should take a few minutes, after which, PFLOTRAN-OGS will be ready to use.
## Testing the PFLOTRAN-OGS Installation - Regression Tests
make test

When running the regression tests, the screen will look like this:

Running pflotran regression tests :

--------------------------------------------------------------------------------
Regression test summary:
    Total run time: 106.924 [s]
    Total tests : 148
    Tests run : 148
    All tests passed.
	
After running the tests, if everything went fine, we can clean up, which includes deleting the log files, with:
```
make clean-tests	
make clean-tests && make test
```
## Running PFLOTRAN-OGS From the Command Line
Through Scripts

Directly Through the Command Line
```
/home/myusername/petsc/ubuntu-opt/bin/mpirun -np 4 /home/myusername/pflotran/src/pflotran -pflotranin spe10.in -output_prefix test_spe10_run
```
# Advanced PETSc Options
Configuring against existing Open MPI Install

Here is an example:
```
CONFIGURE_OPTIONS = --with-debugging=0 --download-fblaslapack=1 --with-fc=/usr/lib64/openmpi/bin/mpif90 --with-cc=/usr/lib64/openmpi/bin/mpicc --with-cxx=/usr/lib64/openmpi/bin/mpicxx --with-mpi-include=/usr/include/openmpi-x86_64 --with-mpi-lib=/usr/lib64/openmpi/lib/libmpi.so --download-cmake=1 --download-ptscotch=1 -download-hypre=1 --download-hdf5=1 --with-c2html=0 COPTFLAGS=-O3 CXXOPTFLAGS=-O3 FOPTFLAGS=-O3 --with-shared-libraries=0
```

Note that it might be necessary to ensure certain libraries are in path, e.g.

export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH

Recall that you must use the mpirun binary associted with the Open MPI install for running PFLOTRAN-OGS, so the above example of running on the command line will generalizes to:
```
/location/of/correct/mpirun -np 4 /home/myusername/pflotran/src/pflotran -pflotranin spe10.in -output_prefix test_spe10_run
```