# install MM-PIHM

make cvode

make [model]

The [model] should be replaced by the name of model that you want to compile, which could be pihm, flux-pihm, flux-pihm-bgc, or rt-flux-pihm.

## clean the executables and object files

make clean

## deep groundwater module (DGW)

make DGW=on [model]

## Build parallelized CVODE

make CVODE_OMP=on [model]

make DEBUG=off [model]

export OMP_NUM_THREADS=12


