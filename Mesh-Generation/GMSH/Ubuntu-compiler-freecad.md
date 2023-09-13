# ubuntu 22.04 编译安装 FreeCAD

## 安装依赖项

sudo apt install cmake cmake-qt-gui libcoin-dev libeigen3-dev libgts-bin libgts-dev libkdtree++-dev libmedc-dev libopencv-dev libproj-dev libpyside2-dev libshiboken2-dev libspnav-dev libx11-dev libxerces-c-dev libzipios++-dev occt-draw pyside2-tools python3-dev python3-matplotlib python3-pivy python3-ply python3-pyside2.qtcore python3-pyside2.qtgui python3-pyside2.qtsvg python3-pyside2.qtwidgets swig

## 其他依赖项（已经编译安装）

libqt5opengl5-dev libqt5svg5-dev libqt5webkit5-dev libqt5x11extras5-dev libqt5xmlpatterns5-dev qtbase5-dev qttools5-dev libvtk7-dev 
libocct-data-exchange-dev libocct-ocaf-dev libocct-visualization-dev 


libboost-date-time-dev libboost-dev libboost-filesystem-dev libboost-graph-dev libboost-iostreams-dev libboost-program-options-dev libboost-python-dev libboost-regex-dev libboost-serialization-dev libboost-thread-dev 

## PySide的安装

wget -c http://ftp.at.debian.org/debian/pool/main/p/pyside2/python3-pyside2uic_5.11.2-3_all.deb
sudo dpkg -i ./python3-pyside2uic_5.11.2-3_all.deb 

## 编译安装FreeCAD

下载源码并切换到最近稳定版本: https://github.com/FreeCAD/FreeCAD.git

mkdir build && cd build
cmake .. -DBUILD_QT5=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 .. && make -j
sudo make install

## 参考

    Ubuntu20.10系统FreeCAD 0.19编译安装
    Compile on Linux


