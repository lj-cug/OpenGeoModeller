# 3.1. Preparing Working Environment
export PROGS=/home/lijian/path-to-regesm

## netcdf
export LD_LIBRARY_PATH=$PROGS/netcdf-4.3.0/lib:$LD_LIBRARY_PATH
export NETCDF=$PROGS/netcdf-4.3.0
export PATH=$NETCDF/bin:$PATH
## parallel-netcdf
export PNETCDF=$PROGS/parallel-netcdf-1.3.1

## llvm and omesa (off-screen rendering by CPU)
export LD_LIBRARY_PATH=$PROGS/llvm-3.9.1/lib:$LD_LIBRARY_PATH
export MESA_INSTALL_PREFIX=$PROGS/mesa-17.0.0

# 3.2 Installation of Model Components
## As it can be seen from the figures/ch03_fig01.png, each model components (including "driver" itself, which is called as "drv" in the figure) use its own directory for the source files, input and output. In addition to the directory structure, the configuration files, run script (OpenPBS, LSF etc.), input and output files can be placed in the main working directory (**BASE_DIR**). RegESM executable (**DRV_SRC**) placed in the working directory is the soft link and can be created with following commands, 
cd $BASE_DIR
ln -s $DRV_SRC/regesm.x




