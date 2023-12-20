# Ubuntu 18.04 apt-get install安装的默认路径

```
include_path: /usr/include/  
lib_path: /usr/lib/x86_64-linux-gnu/
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
```

# MPI

在电脑上分别编译和安装了openmpi-3.1和mpich2-3.4 (3.3),发现两者有区别：

mpich2-3.4的lib中，有libmpi.so.12; 

openmpi的lib中，是libmpi.so.40; 

另外还有很多mpich2没有的so文件，例如libmpi_mpifh.so等，很多是编译ESMF_7.1所需要的。

**建议使用openmpi-3.1**

## 设置MPI库的环境变量

```
export MPI_ROOT=/home/lijian/openmpi_install
export PATH=$MPI_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$MPI_ROOT/lib:$LD_LIBRARY_PATH
export CPATH=$MPI_ROOT/lib:$CPATH
```

## xerces-c-3.2.4安装

```
tar -zxvf xerces-c-3.2.4.tar.gz
cd xerces-c-3.2.4
mkdir build
cd build
cmake ../
make
make test
sudo make install
```

