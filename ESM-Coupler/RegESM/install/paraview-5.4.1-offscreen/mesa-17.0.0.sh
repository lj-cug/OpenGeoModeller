# build llvm-3.9.1
cd $PROGS
wget http://releases.llvm.org/3.9.1/llvm-3.9.1.src.tar.xz
tar -xvf llvm-3.9.1.src.tar.xz
mv llvm-3.9.1.src llvm-3.9.1
cd llvm-3.9.1
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PROGS/llvm-3.9.1 -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_INSTALL_UTILS=ON -DLLVM_TARGETS_TO_BUILD:STRING=X86 -DCMAKE_CXX_FLAGS="-std=c++11" -DBUILD_SHARED_LIBS=ON ../
make
make install
##rm $PROGS/llvm-3.9.1.src.tar.xz

# build mesa-17.0.0
cd $PROGS
wget https://mesa.freedesktop.org/archive/mesa-17.0.0.tar.gz
tar -zxvf mesa-17.0.0.tar.gz
cd mesa-17.0.0/
mkdir build
mv * build/.
mv build src
cd src
./configure --prefix=$PROGS/mesa-17.0.0 --enable-opengl --disable-gles1 --disable-gles2 --disable-va --disable-xvmc --disable-vdpau --enable-shared-glapi --disable-texture-float --enable-gallium-llvm --enable-llvm-shared-libs --with-gallium-drivers=swrast,swr --disable-dri --with-dri-drivers= --disable-egl --with-egl-platforms= --disable-gbm --disable-glx --disable-osmesa --enable-gallium-osmesa --with-llvm-prefix=$PROGS/llvm-3.9.1/build
make
make install
##rm $PROGS/mesa-17.0.0.tar.gz

