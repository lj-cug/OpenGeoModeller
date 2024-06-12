# install foam2csr
����װ, ����OpenFOAM-v2212���˲���

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
    -I${PETSC_INC} \    # �ϱߵ�2�и�ɾ����,�������Ӳ���PETSc��ͷ�ļ�
    -I${AMGX_INC} \
    -I${SPECTRUM_MPI_HOME}/include

LIB_LIBS = \
    -lfiniteVolume \
    -lmeshTools \
    $(foreach dir,$(PETSC_LIB_DIR),-L$(dir)) -lpetsc \
    -L$(AMGX_LIB) -lamgxsh
```

# install external-solver-amgxwrapper
ע�⣺ https://github.com/maorz1998/amgx4foam   
������ǹٷ�����, ����˵����openfoam-7����, ����ʹ��openfoam-9����ʧ��, ��Ҫ��һ������.

ʹ�ã� https://develop.openfoam.com/modules/external-solver/-/tree/amgxwrapper?ref_type=heads

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
    $(foreach dir,$(PETSC_INC_DIR),-I$(dir)) \     # ��PETSC_INC_DIR�޸�ΪPETSC_INC
    -I$(AMGX_INC) \
    -I$(FOAM2CSR_INC)

LIB_LIBS = \
    -lfiniteVolume \
    -lmeshTools \
    $(foreach dir,$(PETSC_LIB_DIR),-L$(dir)) -lpetsc \
    -L$(AMGX_LIB) -lamgx \
    -L$(FOAM_MODULE_LIBBIN) -lfoam2csr
```