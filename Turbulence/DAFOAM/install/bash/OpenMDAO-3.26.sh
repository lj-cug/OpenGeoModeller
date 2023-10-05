# openmdao
. $HOME/dafoam/loadDAFoam.sh && \
pip install openmdao==3.26

# Mphys
. $HOME/dafoam/loadDAFoam.sh && \
cd $HOME/dafoam/repos && \
wget https://github.com/OpenMDAO/mphys/archive/b6db107d05937d95a46b392b6f5759677a93e46d.tar.gz -O mphys.tar.gz && \
tar -xvf mphys.tar.gz && mv mphys-* mphys && \
cd mphys && pip install -e .

# FUNtoFEM-0.3
. $HOME/dafoam/loadDAFoam.sh && \
cd $HOME/dafoam/repos && \
wget https://github.com/smdogroup/funtofem/archive/refs/tags/v0.3.tar.gz -O funtofem.tar.gz && \
tar -xvf funtofem.tar.gz && mv funtofem-* funtofem && \
cd funtofem && cp Makefile.in.info Makefile.in && \
sed -i "s/git/dafoam\/repos/g" Makefile.in && \
make && pip install -e .

# TACS-3.2.1
. $HOME/dafoam/loadDAFoam.sh && \
cd $HOME/dafoam/repos && \
wget https://github.com/smdogroup/tacs/archive/refs/tags/v3.2.1.tar.gz -O tacs.tar.gz && \
tar -xvf tacs.tar.gz && mv tacs-* tacs && \
cd tacs/extern && \
wget https://github.com/DAFoam/files/releases/download/TACS_Extern/TACS_extern.tar.gz && tar -xzf TACS_extern.tar.gz && \
rm -rf metis-4.0.3* && \
wget https://github.com/DAFoam/files/releases/download/TACS_Extern/metis-5.1.0.tar.gz && \
tar -czvf TACS_extern.tar.gz metis*.tar.gz UFconfig*.tar.gz AMD*.tar.gz &&\
tar -xzf metis*.tar.gz && \
cd metis-5.1.0 && make config prefix=$HOME/dafoam/repos/tacs/extern/metis/ CFLAGS="-fPIC" && make install && \
cd ../../ && \
cp Makefile.in.info Makefile.in && \
ls && \
sed -i "s/git/dafoam\/repos/g" Makefile.in && \
make && pip install -e . && \
cd extern/f5tovtk && make && cp f5tovtk $HOME/dafoam/OpenFOAM/sharedBins
