git clone --branch 1.17 https://github.com/hfp/libxsmm
cd libxsmm
make generator
cp bin/libxsmm_gemm_generator $HOME/bin
cd ..