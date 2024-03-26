# This library is mainly required for ESMF installation. It is responsible to read/write grid definitions and attributes (field, component and state level) in XML format.
cd $PROGS
wget http://apache.bilkent.edu.tr//xerces/c/3/sources/xerces-c-3.1.1.tar.gz tar -zxvf xerces-c-3.1.1.tar.gz
cd xerces-c-3.1.1
./configure --prefix=$PROGS/xerces-c-3.1.1 CC=gcc CXX=g++
make
make install