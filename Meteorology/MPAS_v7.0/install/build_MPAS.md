# build MPAS

MPAS-v4.0和MPAS-v7.0, 使用gcc/gfortran V7.5成功编译

修改了v7.0中的mpas_io.F 的两处参数值, 因为PIO和PIO2缺少这几个参数?

为了解决Ubuntu的gcc编译后，ld链接的问题,
在Makefile中的gfortran 编译参数部分，添加了 -fPIC  参数:

"CFLAGS_OPT = -O3 -m64 -fPIC" \
