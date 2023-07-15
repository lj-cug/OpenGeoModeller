wget https://ftp.mcs.anl.gov/pub/pdetools/spack-pkgs/parmetis-4.0.3.tar.gz
tar -xvf parmetis-4.0.3.tar.gz
cd parmetis-4.0.3
#edit ./metis/include/metis.h IDXTYPEWIDTH to be 64 (default is 32).
make config cc=mpicc cxx=mpiCC prefix=$HOME
make install
cp build/Linux-x86_64/libmetis/libmetis.a $HOME/lib
cp metis/include/metis.h $HOME/include
cd ..

# (Make sure $HOME/include contains metis.h and $HOME/lib contains libmetis.a. Otherwise, compile error: cannot find parmetis.)