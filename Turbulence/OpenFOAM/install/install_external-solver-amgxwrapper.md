# install foam2csr
本安装, 基于OpenFOAM-v2212做了测试

## set env. variables
```
export PETSC_INC=
export AMGX_INC=
export PETSC_LIB_DIR=
export AMGX_LIB=
```

See:
```
EXE_INC = \
    -I${PETSC_INC} \    # 上边的2行给删掉了,否则链接不上PETSc的头文件
    -I${AMGX_INC} \
    -I${SPECTRUM_MPI_HOME}/include

LIB_LIBS = \
    -lfiniteVolume \
    -lmeshTools \
    $(foreach dir,$(PETSC_LIB_DIR),-L$(dir)) -lpetsc \
    -L$(AMGX_LIB) -lamgxsh
```

# install external-solver-amgxwrapper
注意： https://github.com/maorz1998/amgx4foam   
这个不是官方代码, 作者说可用openfoam-7编译, 但我使用openfoam-9编译失败, 需要进一步测试.

使用： https://develop.openfoam.com/modules/external-solver/-/tree/amgxwrapper?ref_type=heads

## set env. variables
```
export PETSC_INC_DIR=
export AMGX_INC=
export FOAM2CSR_INC=
export AMGX_LIB=
```

See:
```
EXE_INC = \
    $(PFLAGS) $(PINC) \
    -Wno-old-style-cast \
    -I$(LIB_SRC)/finiteVolume/lnInclude \
    -I$(LIB_SRC)/meshTools/lnInclude \
    -I$(LIB_SRC)/Pstream/mpi \
    $(foreach dir,$(PETSC_INC_DIR),-I$(dir)) \     # 把PETSC_INC_DIR修改为PETSC_INC
    -I$(AMGX_INC) \
    -I$(FOAM2CSR_INC)

LIB_LIBS = \
    -lfiniteVolume \
    -lmeshTools \
    $(foreach dir,$(PETSC_LIB_DIR),-L$(dir)) -lpetsc \
    -L$(AMGX_LIB) -lamgx \
    -L$(FOAM_MODULE_LIBBIN) -lfoam2csr
```