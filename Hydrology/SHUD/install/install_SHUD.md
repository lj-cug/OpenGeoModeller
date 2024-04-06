# install libomp5 for Ubuntu 20.04

apt install libomp-dev libomp5-10

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# install shud

.configure

make clean

make shud_omp -j8




