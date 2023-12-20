# ESMF 7 安装

ESMF V7.0  can only be built using gcc and g++ v4.8, failed using v 7.5

ESMF V7.1 was successfully built using gcc and g++ v7.5, and openmpi v3.1 and mpich2-3.3

ESMF V8.0 was successfully built using gcc and g++ v7.5, and openmpi v3.1 and mpich2-3.3 

建议使用openmpi v3.1，这是RegESM需要的

# PIO2安装

使用gcc/gfortran v7.5 编译成功，并设置环境变量： 

export PI2O=/home/lijian/ESM_lj/pio2
