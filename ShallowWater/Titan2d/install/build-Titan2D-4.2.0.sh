Installation instruction from binaries
Download and unpack:
#download
wget https://github.com/TITAN2D/titan2d/releases/download/v4.2.0/titan2d-v4.2.0-Linux-64bit.tar.gz
#untar
tar xvzf titan2d-v4.2.0-Linux-64bit.tar.gz


To test:
#get to example directory
cd titan2d-v4.2.0/share/titan2d_examples/colimafinemini/Coulomb

#execute with 2 threads
../../../../bin/titan -nt 2 input.py
Compiling with help of titan2d_dep, package containing all non-standard dependencies
Provided compiled dependencies package, titan2d_dep.tar.gz, works with CentOS 6/7 and titan2d_dep-Ubuntu18.tar.gz with Ubuntu 18/20.

# download dependencies package
wget https://github.com/TITAN2D/titan2d/releases/download/v4.2.0/titan2d_dep.tar.gz
# in case of Ubuntu (18 or 20)
# wget https://github.com/TITAN2D/titan2d/releases/download/v4.2.0/titan2d_dep-Ubuntu18.tar.gz
tar xvzf titan2d_dep.tar.gz

#download source code package
wget https://github.com/TITAN2D/titan2d/releases/download/v4.2.0/titan2d-v4.2.0-src.tar.gz
tar xvzf titan2d-v4.2.0-src.tar.gz

#compile titan
mkdir titan2d-v4.2.0-bld
cd titan2d-v4.2.0-bld
../titan2d-v4.2.0/configure --prefix=/full/path/to/install --enable-openmp --with-titan2d-dep=/full/path/to/titan2d_dep
#--enable-portable will copy titan2d_dep to installation folder and thus original titan2d_dep can be removed
make -j 4
make install


