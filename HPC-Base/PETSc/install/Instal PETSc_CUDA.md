# Installing PETSc to use NVidia GPUs (aka CUDA)

   Install CUDA, Thrust, Cusp in default locations; /usr/local/cuda. 
   The versions and locations of Thrust and Cusp you should use are listed in $PETSC_DIR/config/PETSc/packages/cuda.py. 
   The required version of CUDA may not be publically available, you may need to register as an NVIDIA Developer (free) to access them. 
   
Make sure nvcc is in PATH

On Linux

install compatible NVidia kernel developer driver by running the executable you download, as root
make sure  LD_LIBRARY_PATH is set to point to the CUDA libraries, for instance
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib

verify CUDA installation with Nvidia CUDA SDK [by compiling the SDK - and then running 'deviceQuery' && 'binomialOptions -type=double'] (how do you do this????)

Configure and build PETSc with the additional configure options
--with-cuda=1 --with-cusp=1 --with-thrust=1

if the GPU card only supports single precision add --with-precision=single
if you did not install in default locations add --with-thrust-dir=path_to_thrust and --with-cusp-dir=path_to_cusp

Run a sample example with:

cd src/snes/examples/tutorials
make ex19
./ex19 -da_vec_type mpicusp -da_mat_type mpiaijcusp -pc_type none -dmmg_nlevels 1 -da_grid_x 100 -da_grid_y 100 -log_summary -mat_no_inode -preload off  -cusp_synchronize

We only have experience using Nvidia GPUs on Apple machines and Intel Xeon servers running Ubuntu 10.04 x86_64 NVidia GT200 [Tesla C1060] 