# ����Paraview-5.8.1��QT������

Paraview��Ҫ��װQt5�������Qt5-5.9��
```
sudo apt-get -y install qt5-default
sudo apt-get -y install qtbase5-dev
sudo apt-get -y install qttools5-dev
apt-get install qtcreator qtbase5-private-dev qt5ct qtdeclarative5-dev qtdeclarative5-dev-tools
```

����, ����Qt5-Linux.run��װ��

export Qt5_DIR=/opt/Qt5.12.12/5.12.12/gcc_64/lib/cmake/Qt5  # ע��·��

## ����Qt5�Ļ�������
```
export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:/opt/Qt5.12.12/5.12.12/gcc_64/lib/cmake/
export PATH=/opt/Qt5.12.12/5.12.12/gcc_64/bin/:${PATH}
export LD_LIBRARY_PATH=/opt/Qt5.12.12/5.12.12/gcc_64/lib/:${LD_LIBRARY_PATH}
```

�����ҪParaview���avi�����ļ�������Ҫ��װFfmpeg�⣬Դ�����أ�

����ڲ�֧��OpenGL�ĵ���������paraview����Ҫ��װMESA 3D��--ģ��Ӳ��������⡣

# ��װVR���
CMAKE���������ӣ�

-DBUILD_SHARED_LIB -DPARAVIEW_BUILD_QT_GUI -DPARAVIEW_USE_MPI  -DPARAVIEW_BUILD_PLUGIN_VRPlugin

��Ҫ��װOpenVR��
``` 
-- Installing: /usr/local/lib/libopenvr_api.a
-- Installing: /usr/local/include/openvr/openvr_driver.h
-- Installing: /usr/local/include/openvr/openvr_capi.h
-- Installing: /usr/local/include/openvr/openvr.h
-- Installing: /usr/local/share/pkgconfig/openvr.pc
```

## ʹ��VRUI�� 
-DPARAVIEW_USE_VRUI   