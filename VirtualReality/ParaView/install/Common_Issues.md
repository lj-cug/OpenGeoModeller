# Common Issues

(1) Qt5Help的问题，解决方法：apt-get -y install qttools5-dev

(2) Target "pqApplicationComponents" links to target ‘Qt5:SVG“ but the target was not found. Perhaps a find_pacage() call is missing for an IMPORTED target, or an ALIAS target is missing?

解决方法：apt-get install libqt5svg5 libqt5svg5-dev

(3) Could not find OpenVR external dependency.

CMAKE-GUI中打开Advanced, 设置OPENVR_INCLUDE_DIR 的路径。

(4) Could not find a package configuration file provided by "Qt5" (requested
  version 5.9) with any of the following names:

    Qt5Config.cmake
	
    qt5-config.cmake
	
下载Qt5-Linux.run安装，然后在cmake-gui中设置 Qt5_DIR = /opt/Qt5.9.9/5.9.9/gcc_64/lib/cmake/Qt5

(5)安装到99%，include"pqvcrtoolbar.h"找不到的问题，是因为头文件名改变了，应该是pqVCRToolbar.h

（6）(1.085s) [paraview        ]  vtkPVPluginLoader.cxx:530    ERR| vtkPVPluginLoader (0x56365f4d30d0): /home/lijian/ParaView/build/lib/paraview-5.8/plugins/VRPlugin/VRPlugin.so: undefined symbol: _ZTV13pqVRDockPanel
??????

使用ldd命令查看一下VRPlugin.so需要的so库，重新编译VRPlugin

(7)
Could NOT find MPI_C (missing: MPI_C_WORKS) 

Could NOT find MPI (missing: MPI_C_FOUND C) 

    Reason given by package: MPI component 'Fortran' was requested, but language Fortran is not enabled. 

解决：使用编译的新版MPI库	
	
export MPI_HOME=/mnt/lijian/3rd-library-install/mpich-3.3

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPI_HOME/lib

export CMAKEFLAGS="${CMAKEFLAGS} -DMPIEXEC_EXECUTABLE=${MPI_HOME}/bin/mpiexec"
