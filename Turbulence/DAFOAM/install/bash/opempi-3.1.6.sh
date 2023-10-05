#
echo '# OpenMPI-3.1.6' >> $HOME/dafoam/loadDAFoam.sh && \
echo 'export MPI_INSTALL_DIR=$DAFOAM_ROOT_PATH/packages/openmpi-3.1.6/opt-gfortran' >> $HOME/dafoam/loadDAFoam.sh && \
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPI_INSTALL_DIR/lib' >> $HOME/dafoam/loadDAFoam.sh && \
echo 'export PATH=$MPI_INSTALL_DIR/bin:$PATH' >> $HOME/dafoam/loadDAFoam.sh&& \
. $HOME/dafoam/loadDAFoam.sh

# install openmpi-3.1.6
cd $HOME/dafoam/packages && \
#wget https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.6.tar.gz  && \
tar -xvf openmpi-3.1.6.tar.gz && \
cd openmpi-3.1.6 && \
./configure --prefix=$MPI_INSTALL_DIR && \
make all install

#mpicc -v

pip install mpi4py==3.1.3
