# Install AmgX

tar xf AMGX-2.1.0.tar.gz

cd AMGX-2.1.0

mkdir build

cd build

cmake --CMAKE_INSTALL_PREFIX=... --CUDA_SDK_ROOT_DIR=/usr/local/cuda --CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda ..


## Caution

Because AmgX used CUDA compilation, CUDA-10.1 must use GNU compiler version lower 8

# Install AmgX_Wrapper (AmgX-PETSc)



# Install AmgX_Wrapper_c


