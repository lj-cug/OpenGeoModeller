# 安装WRF需要的编译器和程序库

export CC=gcc
export CXX=g++
export FC=gfortran
export FCFLAGS="-m64"
export NETCDF=/usr/local/

HDF5 not set in environment. Will configure WRF for use without.
Will use 'time' to report timing information

# 安装Jasper

$JASPERLIB or $JASPERINC not found in environment, configuring to build without grib2 I/O...

安装Jasper，使用CMAKE，安装路径： 

export JASPERINC=/home/lijian/ESM_lj/jasper/include/
export JASPERLIB=home/lijian/ESM_lj/jasper/lib/
export LDFLAGS="-L $JASPERLIB"
export CXXFLAGS="-I $JASPERINC"

出现：Right now you are not getting the Jasper lib, from the environment, compiled into WRF 不管他！

选择用gnu编译器，编译dmpar

选择基本模式，basic    1

生成编译WRF的configuration file      configure.wrf

# 安装WRF_io_esmf

在configure.wrf中添加 ESMF_DIR

使用gcc gfortran v7，以及使用MPICH-3.3 (使用gcc gfortran v7.5编译安装)，成功编译了 WRF V3.9.1 以及WRFV3911 + ESMF.

WRFV3911 + ESMF的main文件夹下，原始有测试WRF与dummy模式的子程序 (wrf_SST_ESMF.F  )，现在在makefile将其注释掉了，
增加了wrf_test_ESMF.F 就是wrf_SST_ESMF.F???


使用 gcc gfortran v7成功编译了 WRFV4.3 (standalone run) 和 WRFV4.3 + ESMF，使用mpich-3.3, 其中直接把ESMF相关的F文件拷贝到相应的位置，io_esmf中的库文件需要手动编译: make -r
然后，WRF就可以链接到libwrfio_esmf.a了!
 