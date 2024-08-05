# gcc, CMake, openmpi

# libjpeg or libjpeg-turbo for (de)compressing the rendered image of the transmission
apt-get install libjpeg-dev

# Jansson for the de- and encryption of the JSON messages transfered between server and client
## git clone https://github.com/akheron/jansson.git
apt-get install libjansson-dev

# Boost(at least 1.56) is needed, but only template libraries, so no system wide installation or static linking is needed
## wget http://sourceforge.net/projects/boost/files/boost/1.56.0/boost_1_56_0.tar.gz/download -O boost_1_56_0.tar.gz 
## ./bootstrap.sh --prefix=$BOOST/install     ./b2     ./b2 install 
apt-get install libboost-dev

# Requirements for the in situ library and the examples using it
## alpaka

## IceT
`
git clone git://public.kitware.com/IceT.git
cmake .. -DCMAKE_INSTALL_PREFIX=../install
## Later while compiling an application using ISAAC (including the examples) add
-DIceT_DIR=$ICET/install/lib
`

# Requirements for the server only (Backend Computation node)
## libwebsockets for the connection between server and an HTML5 client
git clone https://github.com/warmcat/libwebsockets.git

## gStreamer is only needed, if streaming over RTP or the Twitch plugin shall be used.
apt-get install libgstreamer1.0-dev libgstreamer-plugins-base0.10-dev libgstreamer-plugins-good1.0-dev libgstreamer-plugins-bad1.0-dev

# Building
## Installing the library
git clone https://github.com/ComputationalRadiationPhysics/isaac.git
cd isaac
cd lib
mkdir build
cd build
cmake ..
make install

## The example
cd example
mkdir build
cd build
cmake -DIceT_DIR=$ICET/install/lib -DISAAC_ALPAKA=OFF ..
make

# For running these examples you need a running isaac server.
## The server  'Backend computation node'
### The server resides in the directory `server` and also uses CMake:
cd isaac
cd server
mkdir build
cd build
cmake -DLibwebsockets_DIR=$LIBWEBSOCKETS/install/lib/cmake/libwebsockets ..
`
    * `-DISAAC_GST=OFF` → Deactivates GStreamer (Default if not found).
    * `-DISAAC_JPEG=OFF` → Deactivates JPEG compression. As already mentioned: This is not advised
      and will most probably leave ISAAC in an unusable state in the end.
    * `-DISAAC_SDL=ON` → Activates a plugin for showing the oldest not yet finished
      visualization in an extra window using `libSDL`. Of course this option does not
      make much sense for most servers as they don't have a screen or even an
      X server installed.
`
make

# If you want to install the server type. 其实不需要安装
cmake -DCMAKE_INSTALL_PREFIX=/your/path ..
make install

# However, the server doesn't need to be installed and can also directly be called with
./isaac
./isaac --help

# Testing   在服务端启动测试程序
## To test the server and an example, just start the server with `./isaac`, connect to it with one of the HTML clients in the directory `client` (best is `interface.htm`) and start an example with `./example`
## It should connect to the server running on localhost and be observable and steerable. You can run multiple instances of the example with `mpirun -c N ./example` with the number of instances `N`. To exit the example, use the client or ctrl+C.

