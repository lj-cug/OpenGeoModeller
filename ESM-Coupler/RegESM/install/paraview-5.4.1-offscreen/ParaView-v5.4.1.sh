# Option **-DOSMESA_LIBRARY** can be also set as **${MESA_INSTALL_PREFIX}/lib/libOSMesa.a** to use static library rather than dynamic one.
# The performance of COP component can be affected in case of using software emulation for rendering. The initial tests show that the OSMesa installation is 10x slower than EGL configuration. 
# If there is a possibility to have access to X window to use Catalyst Live feature, then there is no need to install ParaView with EGL libraries.
##
cd $PROGS
wget -O ParaView-v5.4.1.tar.gz "https://www.paraview.org/paraview-downloads/download.php?submit=Download&version=v5.4&type=binary&os=Sources&downloadFile=ParaView-v5.4.1.tar.gz"
tar -zxvf ParaView-v5.4.1.tar.gz
mv ParaView-v5.4.1 paraview-5.4.1
cd paraview-5.4.1
mkdir src
mv * src/.
mkdir build
cd build
cmake \
  -DCMAKE_BUILD_TYPE=Release                                   \
  -DPARAVIEW_ENABLE_PYTHON=ON                                  \
  -DPARAVIEW_USE_MPI=ON                                        \
  -DPARAVIEW_BUILD_QT_GUI=OFF                                  \
  -DVTK_USE_X=OFF                                              \
  -DOPENGL_INCLUDE_DIR=IGNORE                                  \
  -DOPENGL_xmesa_INCLUDE_DIR=IGNORE                            \
  -DOPENGL_gl_LIBRARY=IGNORE                                   \
  -DOSMESA_INCLUDE_DIR=${MESA_INSTALL_PREFIX}/include          \
  -DOSMESA_LIBRARY=${MESA_INSTALL_PREFIX}/lib/libOSMesa.so     \
  -DVTK_OPENGL_HAS_OSMESA=ON                                   \
  -DVTK_USE_OFFSCREEN=OFF ../src
make
##rm $PROGS/ParaView-v5.4.1.tar.gz

