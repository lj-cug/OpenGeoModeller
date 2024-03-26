# This step is optional but if there is a plan to use ESMF NetCDF I/O capabilities to write exchange fields to disk through the use of [parallel-netcdf](https://trac.mcs.anl.gov/projects/parallel-netcdf) library (1.3.1) is required.
#
cd $PROGS
wget http://ftp.mcs.anl.gov/pub/parallel-netcdf/parallel-netcdf-1.3.1.tar.gz
tar -zxvf parallel-netcdf-1.3.1.tar.gz
cd parallel-netcdf-1.3.1
./configure --prefix=`pwd` --with-mpi=/opt/openmpi/1.6.5/intel/2013psm FC=mpif90 F77=mpif90 CXX=mpiccpc
make -j8
make install

