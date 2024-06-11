# install foam2csr
## 设置环境变量
```
EXE_INC = \
    -I. \
    -I$(CUBROOT) \
    -I${PETSC_INC} \
    -I${AMGX_INC} \
    -I${SPECTRUM_MPI_HOME}/include

LIB_LIBS = \
    -lfiniteVolume \
    -lmeshTools \
    $(foreach dir,$(PETSC_LIB_DIR),-L$(dir)) -lpetsc \
    -L$(AMGX_LIB) -lamgxsh
```

# install amgx4foam
还要事先安装好petsc4foam库

## 设置环境变量
```
-I/usr/local/cuda-11.6/include

AMGX_INC
AMGX_LIB

FOAM2CSR_INC
FOAM2CSR_LIB
```




