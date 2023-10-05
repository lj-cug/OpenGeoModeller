. $HOME/dafoam/loadDAFoam.sh && \
cd $HOME/dafoam/repos && \
wget https://github.com/mdolab/baseclasses/archive/v1.6.1.tar.gz -O baseclasses.tar.gz && \
tar -xvf baseclasses.tar.gz && cd baseclasses-1.6.1 && pip install . && \
cd $HOME/dafoam/repos && \
wget https://github.com/mdolab/pyspline/archive/v1.5.2.tar.gz -O pyspline.tar.gz && \
tar -xvf pyspline.tar.gz && cd pyspline-1.5.2 && \
cp config/defaults/config.LINUX_GFORTRAN.mk config/config.mk && \
make && pip install . && \
cd $HOME/dafoam/repos && \
wget https://github.com/mdolab/pygeo/archive/4641625e28b0bae97e05351c23d9083a693734d3.tar.gz -O pygeo.tar.gz && \
tar -xvf pygeo.tar.gz && mv pygeo-* pygeo && cd pygeo && pip install . && \
cd $HOME/dafoam/repos && \
wget https://github.com/mdolab/multipoint/archive/v1.4.0.tar.gz -O multipoint.tar.gz && \
tar -xvf multipoint.tar.gz && cd multipoint-1.4.0 && pip install . && \
cd $HOME/dafoam/repos && \
wget https://github.com/mdolab/pyhyp/archive/v2.5.0.tar.gz -O pyhyp.tar.gz && \
tar -xvf pyhyp.tar.gz && cd pyhyp-2.5.0 && \
cp -r config/defaults/config.LINUX_GFORTRAN_OPENMPI.mk config/config.mk && \
make && pip install . && \
cd $HOME/dafoam/repos && \
wget https://github.com/mdolab/cgnsutilities/archive/v2.6.0.tar.gz -O cgnsutilities.tar.gz && \
tar -xvf cgnsutilities.tar.gz && cd cgnsutilities-2.6.0 && \
cp config/defaults/config.LINUX_GFORTRAN.mk config/config.mk && \
make && pip install . && \
cd $HOME/dafoam/repos && \
wget https://github.com/mdolab/idwarp/archive/v2.6.0.tar.gz -O idwarp.tar.gz && \
tar -xvf idwarp.tar.gz && cd idwarp-2.6.0 && \
cp -r config/defaults/config.LINUX_GFORTRAN_OPENMPI.mk config/config.mk && \
make && pip install . && \
cd $HOME/dafoam/repos && \
wget https://github.com/mdolab/pyoptsparse/archive/v2.10.1.tar.gz -O pyoptsparse.tar.gz && \
tar -xvf pyoptsparse.tar.gz && cd pyoptsparse-2.10.1 && \
pip install .
