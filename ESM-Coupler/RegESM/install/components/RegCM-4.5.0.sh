# Regional Climate Model (RegCM)
## version > 4.5; need to be patched for co-processing

mkdir $ATM_SRC
cd $ATM_SRC
#wget https://gforge.ictp.it/gf/download/frsrelease/252/1580/RegCM-4.5.0.tar.gz
tar -zxvf RegCM-4.5.0.tar.gz
cd RegCM-4.5.0
./bootstrap.sh
./configure --prefix=`pwd` --enable-cpl CC=icc FC=ifort
make
make install
