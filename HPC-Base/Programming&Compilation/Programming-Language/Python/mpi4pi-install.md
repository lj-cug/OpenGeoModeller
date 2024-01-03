# Install mpi4py

�����Ѿ�����ʹ��MPI-enabled Python interpreter���ַ�ʽ��, �����Ҫʹ��, �ɲο���
https://mpi4py.readthedocs.io/en/stable/appendix.html#building-mpi

���ڶ���ʹ��Python���������Ӿ�̬�Ͷ�̬���ӿ�ķ�ʽ.

��Դ�����MPI���mpi4py, ��ִ�����沽��.

## Building MPI from sources

### MPICH
```
$ tar -zxf mpich-X.X.X.tar.gz
$ cd mpich-X.X.X
$ ./configure --enable-shared --prefix=/usr/local/mpich
$ make
$ make install``
```

### Open MPI
```
$ tar -zxf openmpi-X.X.X tar.gz
$ cd openmpi-X.X.X
$ ./configure --prefix=/usr/local/openmpi
$ make all
$ make install
```

### ����LD_LIBRARY_PATH��������

MPICH��
```
MPI_DIR=/usr/local/mpich
export LD_LIBRARY_PATH=$MPI_DIR/lib:$LD_LIBRARY_PATH
```

Open MPI:
```
MPI_DIR=/usr/local/openmpi
export LD_LIBRARY_PATH=$MPI_DIR/lib:$LD_LIBRARY_PATH
```

## installation of mpi4py
### Requirements
```
MPI 
Python 3.5+
```
### Using pip
```
python -m pip install mpi4py
or
env MPICC=/path/to/mpicc python -m pip install mpi4py
```
������°�װmpi4py, ��Ҫ�����֮ǰ��װ���µĻ���, ����ִ�У�
```
python -m pip cache remove mpi4py
or
python -m pip install --no-cache-dir mpi4py
```

### Using distutils
```
wget https://github.com/mpi4py/mpi4py/releases/download/X.Y.Z/mpi4py-X.Y.Z.tar.gz
tar -zxf mpi4py-X.Y.Z.tar.gz
cd mpi4py-X.Y.Z
python setup.py build --mpicc=/where/you/have/mpicc
```
����, ʹ��mpi.cfg
```
mpi.cfg:

[mpi]

include_dirs         = /usr/local/mpi/include
libraries            = mpi
library_dirs         = /usr/local/mpi/lib
runtime_library_dirs = /usr/local/mpi/lib

[other_mpi]

include_dirs         = /opt/mpi/include ...
libraries            = mpi ...
library_dirs         = /opt/mpi/lib ...
runtime_library_dirs = /op/mpi/lib ...

```
Ȼ��, ִ�У�

python setup.py build --mpi=other_mpi

## Testing

mpiexec -n 5 python -m mpi4py.bench helloworld

mpiexec -n 5 python demo/helloworld.py
