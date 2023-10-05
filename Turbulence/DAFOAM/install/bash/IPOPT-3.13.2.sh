#
echo '# Ipopt' >> $HOME/dafoam/loadDAFoam.sh && \
echo 'export IPOPT_DIR=$DAFOAM_ROOT_PATH/packages/Ipopt' >> $HOME/dafoam/loadDAFoam.sh && \
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$IPOPT_DIR/lib' >> $HOME/dafoam/loadDAFoam.sh && \
. $HOME/dafoam/loadDAFoam.sh


#
cd $HOME/dafoam/packages && \
git clone -b stable/3.13 https://github.com/coin-or/Ipopt.git && \
cd $IPOPT_DIR && \
git clone -b stable/2.1 https://github.com/coin-or-tools/ThirdParty-Mumps.git && \
cd ThirdParty-Mumps && \
./get.Mumps && \
./configure --prefix=$IPOPT_DIR && \
make && \
make install

# compile Ipopt
cd $IPOPT_DIR && \
mkdir build && \
cd build && \
../configure --prefix=${IPOPT_DIR} --disable-java --with-mumps --with-mumps-lflags="-L${IPOPT_DIR}/lib -lcoinmumps" --with-mumps-cflags="-I${IPOPT_DIR}/include/coin-or/mumps" && \
make && \
make install
