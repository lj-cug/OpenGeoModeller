# install foam2csr
## ���û�������
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
��Ҫ���Ȱ�װ��petsc4foam��

## ���û�������
```
-I/usr/local/cuda-11.6/include

AMGX_INC
AMGX_LIB

FOAM2CSR_INC
FOAM2CSR_LIB
```




