# 
echo '# CGNS-4.1.2' >> $HOME/dafoam/loadDAFoam.sh && \
echo 'export CGNS_HOME=$DAFOAM_ROOT_PATH/packages/CGNS-4.1.2/opt-gfortran' >> $HOME/dafoam/loadDAFoam.sh && \
echo 'export PATH=$PATH:$CGNS_HOME/bin' >> $HOME/dafoam/loadDAFoam.sh && \
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CGNS_HOME/lib' >> $HOME/dafoam/loadDAFoam.sh && \
. $HOME/dafoam/loadDAFoam.sh


#
cd $HOME/dafoam/packages && \
wget https://github.com/CGNS/CGNS/archive/v4.1.2.tar.gz  && \
tar -xvaf v4.1.2.tar.gz && \
cd CGNS-4.1.2 && \
mkdir -p build && \
cd build && \
cmake .. -DCGNS_ENABLE_FORTRAN=1 -DCMAKE_INSTALL_PREFIX=$CGNS_HOME -DCGNS_BUILD_CGNSTOOLS=0 && \
make all -j8
make install

