#
echo '# Petsc-3.14.6' >> $HOME/dafoam/loadDAFoam.sh && \
echo 'export PETSC_DIR=$DAFOAM_ROOT_PATH/packages/petsc-3.14.6' >> $HOME/dafoam/loadDAFoam.sh && \
echo 'export PETSC_ARCH=real-opt' >> $HOME/dafoam/loadDAFoam.sh && \
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PETSC_DIR/$PETSC_ARCH/lib' >> $HOME/dafoam/loadDAFoam.sh && \
echo 'export PETSC_LIB=$PETSC_DIR/$PETSC_ARCH/lib' >> $HOME/dafoam/loadDAFoam.sh && \
. $HOME/dafoam/loadDAFoam.sh

# install
cd $HOME/dafoam/packages && \
wget https://www.mcs.anl.gov/petsc/mirror/release-snapshots/petsc-3.14.6.tar.gz  && \

tar -xvf petsc-3.14.6.tar.gz && \
cd petsc-3.14.6 && \
./configure --PETSC_ARCH=real-opt --with-scalar-type=real --with-debugging=0 --download-metis=yes --download-parmetis=yes --download-superlu_dist=yes --download-fblaslapack=yes --with-shared-libraries=yes --with-fortran-bindings=1 --with-cxx-dialect=C++11 && \
make PETSC_DIR=$HOME/dafoam/packages/petsc-3.14.6 PETSC_ARCH=real-opt all

# install petsc4py-3.14.1
cd $PETSC_DIR/src/binding/petsc4py && \
pip install .

