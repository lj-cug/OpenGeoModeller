conda install meson ninja   # use conda environment, which is same with dafoam

# conda env. for su2 including MPICH
# SU2 needs MPICH2, cannot use openmpi-3 or openmpi-4
export PATH=/root/miniconda3/envs/su2/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/miniconda3/envs/su2/lib
export PYTHONUSERBASE=no-local-libs
export PKG_CONFIG_PATH=/root/miniconda3/envs/su2/lib/pkgconfig:$PKG_CONFIG_PATH
# include <mpi.h>
export C_INCLUDE_PATH=/root/miniconda3/envs/su2/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/root/miniconda3/envs/su2/include:$CPLUS_INCLUDE_PATH

# simple configure
./meson.py build --reconfigure -Denable-autodiff=true -Dcustom-mpi=true -Dextra-deps=mpich -Denable-cgns=true -Denable-tecio=true --prefix=/home/lijian/SU2-8.0.0

./ninja -C build install


#-------------------------------------
# Python wrapper
apt-get install swig
pip install mpi4py

# configure with necessary options
./meson.py build --reconfigure -Denable-autodiff=true -Denable-directdiff=true -Dcustom-mpi=true -Dextra-deps=mpich -Denable-cgns=true -Denable-tecio=true -Denable-pywrapper=true -Denable-openblas=true --prefix=/home/lijian/SU2-8.0.0

./ninja -C build install
