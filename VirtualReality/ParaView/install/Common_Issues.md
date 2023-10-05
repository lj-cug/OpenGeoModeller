# Common Issues

(1) Qt5Help�����⣬���������apt-get -y install qttools5-dev

(2) Target "pqApplicationComponents" links to target ��Qt5:SVG�� but the target was not found. Perhaps a find_pacage() call is missing for an IMPORTED target, or an ALIAS target is missing?

���������apt-get install libqt5svg5 libqt5svg5-dev

(3) Could not find OpenVR external dependency.

CMAKE-GUI�д�Advanced, ����OPENVR_INCLUDE_DIR ��·����

(4) Could not find a package configuration file provided by "Qt5" (requested
  version 5.9) with any of the following names:

    Qt5Config.cmake
	
    qt5-config.cmake
	
����Qt5-Linux.run��װ��Ȼ����cmake-gui������ Qt5_DIR = /opt/Qt5.9.9/5.9.9/gcc_64/lib/cmake/Qt5

(5)��װ��99%��include"pqvcrtoolbar.h"�Ҳ��������⣬����Ϊͷ�ļ����ı��ˣ�Ӧ����pqVCRToolbar.h

��6��(1.085s) [paraview        ]  vtkPVPluginLoader.cxx:530    ERR| vtkPVPluginLoader (0x56365f4d30d0): /home/lijian/ParaView/build/lib/paraview-5.8/plugins/VRPlugin/VRPlugin.so: undefined symbol: _ZTV13pqVRDockPanel
??????

ʹ��ldd����鿴һ��VRPlugin.so��Ҫ��so�⣬���±���VRPlugin

(7)
Could NOT find MPI_C (missing: MPI_C_WORKS) 

Could NOT find MPI (missing: MPI_C_FOUND C) 

    Reason given by package: MPI component 'Fortran' was requested, but language Fortran is not enabled. 

�����ʹ�ñ�����°�MPI��	
	
export MPI_HOME=/mnt/lijian/3rd-library-install/mpich-3.3

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPI_HOME/lib

export CMAKEFLAGS="${CMAKEFLAGS} -DMPIEXEC_EXECUTABLE=${MPI_HOME}/bin/mpiexec"
