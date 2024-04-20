# add this to ~/.bashrc

```
export WRF_LIBS_DIR=$HOME/Build_WRF/LIBRARIES
export LD_LIBRARY_PATH="$WRF_LIBS_DIR/grib2/lib:$WRF_LIBS_DIR/netcdf/lib:$WRF_LIBS_DIR/hdf5/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$WRF_LIBS_DIR/grib2/lib:$WRF_LIBS_DIR/netcdf/lib:$WRF_LIBS_DIR/hdf5/lib:$LIBRARY_PATH"
export C_INCLUDE_PATH="$WRF_LIBS_DIR/grib2/include:$WRF_LIBS_DIR/netcdf/include:$WRF_LIBS_DIR/hdf5/include:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="$WRF_LIBS_DIR/grib2/include:$WRF_LIBS_DIR/netcdf/include:$WRF_LIBS_DIR/hdf5/include:$CPLUS_INCLUDE_PATH"
export PATH=$WRF_LIBS_DIR/netcdf/bin:$PATH
export NETCDF=$WRF_LIBS_DIR/netcdf
export JASPERLIB=$WRF_LIBS_DIR/grib2/lib
export JASPERINC=$WRF_LIBS_DIR/grib2/include

source ${HOME}/.bashrc
```

# setup
```
export CC=gcc
export CXX=g++
export FC=gfortran
export FCFLAGS="-m64 $FCFLAGS"
export F90FLAGS="-fPIC $F90FLAGS"
export F77=gfortran
export FFLAGS="-m64 $FFLAGS"
# export LDFLAGS="-lnetcdf $LDFLAGS"
# export CPPFLAGS="-I$WRF_LIBS_DIR/grib2/include $CPPFLAGS"
export THREADS=$(nproc)
```

# some 3rd parties for WRF4 installation
```
wget https://www2.mmm.ucar.edu/wrf/OnLineTutorial/compile_tutorial/tar_files/zlib-1.2.11.tar.gz
wget https://www2.mmm.ucar.edu/wrf/OnLineTutorial/compile_tutorial/tar_files/netcdf-c-4.7.2.tar.gz
wget https://www2.mmm.ucar.edu/wrf/OnLineTutorial/compile_tutorial/tar_files/netcdf-fortran-4.5.2.tar.gz
wget https://www2.mmm.ucar.edu/wrf/OnLineTutorial/compile_tutorial/tar_files/jasper-1.900.1.tar.gz
wget https://www2.mmm.ucar.edu/wrf/OnLineTutorial/compile_tutorial/tar_files/libpng-1.2.50.tar.gz
wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.13/hdf5-1.13.3/src/hdf5-1.13.3.tar.gz
```

# install 3r parties
```
export THREADS=$(nproc)

tar xvzf hdf5-1.13.3.tar.gz
cd hdf5-1.13.3
./configure --prefix=$DIR/hdf5 --enable-fortran --enable-cxx
make -j $THREADS
make install
cd ..

tar xzvf netcdf-c-4.7.2.tar.gz
cd netcdf-c-4.7.2
#./configure --prefix=$WRF_LIBS_DIR/netcdf --disable-dap --enable-fortran --with-hdf5=$WRF_LIBS_DIR/hdf5
./configure --prefix=$WRF_LIBS_DIR/netcdf --disable-dap CPPFLAGS=-I$WRF_LIBS_DIR/hdf5/include LDFLAGS=-L$WRF_LIBS_DIR/hdf5/lib
make -j $THREADS
make install
cd ..

tar xzvf netcdf-fortran-4.5.2.tar.gz
cd netcdf-fortran-4.5.2
# LDFLAGS="-lnetcdf $LDFLAGS" 
./configure --prefix=$DIR/netcdf CFLAGS="-I$WRF_LIBS_DIR/hdf5/include -I$WRF_LIBS_DIR/netcdf/include" CPPFLAGS="-I$WRF_LIBS_DIR/hdf5/include -I$WRF_LIBS_DIR/netcdf/include" LDFLAGS="-L$WRF_LIBS_DIR/hdf5/lib -L$WRF_LIBS_DIR/netcdf/lib"
make
make install
cd ..

tar xzvf zlib-1.2.11.tar.gz
cd zlib-1.2.11
./configure --prefix=$DIR/grib2
make -j $THREADS
make install
cd ..

tar xzvf libpng-1.2.50.tar.gz
cd libpng-1.2.50
./configure --prefix=$DIR/grib2
make -j $THREADS
make install
cd ..

tar xzvf jasper-1.900.1.tar.gz
cd jasper-1.900.1
./configure --prefix=$DIR/grib2
make -j $THREADS
make install
cd ..
```

# start to install WRF/WRFPLUS/WRFDA
## WRF
```
wget https://github.com/wrf-model/WRF/releases/download/v4.5/v4.5.tar.gz
tar -xvzf v4.5.tar.gz
cp -r WRFV4.5 WRFPLUSV4.5
cp -r WRFV4.5 WRFDAV4.5

cd WRFV4.5
./configure
./compile -j $(nproc) em_real 2>&1 | tee log.compile
ls -ls main/*.exe
cd ..
```

## WRFPLUS
```
cd WRFPLUSV4.5
./configure wrfplus
./compile -j16 2>&1 wrfplus | tee log.compile
ls -ls main/*.exe
export WRFPLUS_DIR=$HOME/Build_WRF/WRFPLUSV4.5
cd ..
```

## WRFDA
```
cd WRFDAV4.5
./configure 4dvar
./compile -j16 all_wrfvar 2>&1 | tee log.compile
./compile -j16 all_wrfvar 2>&1 | tee log.1.compile
ls -ls var/build/*.exe var/obsproc/*.exe
cd ..
```

## ��������

```
���ִ���ɾ������+�����ļ���ʹ�� ./clean -a

1 Symbol ��nf_netcdf4�� at (1) has no IMPLICIT type; did you mean ��nf_cdf5��?
��������ʽ���룬��Ҫ�������� WRF �Ĺٷ��̳��ߣ�����װ���˵�ǰ�汾�� WRFDA
���� netcdf-fortran �����ֱ�� make check ��У���ܷ��ҵ�

2 checking for Fortran flag to compile .f90 files... unknown
��� export F90FLAGS="-fPIC $F90FLAGS"
�ο���Bug�������_tutuerwang�Ĳ���-CSDN����

3 Fatal Error: Cannot open module file ��da_control.mod�� for reading at (1): No such file or directory
������ WRFDA �ı���ű�û��д��������ϵ�����¶��̱߳���ʱ�������⣬����ʱ���� da_control.mod �ᷢ����ʵ���е�
```

