# 编译Paraview-5.8.1的QT库设置

Paraview需要安装Qt5，最好是Qt5-5.9：
```
sudo apt-get -y install qt5-default
sudo apt-get -y install qtbase5-dev
sudo apt-get -y install qttools5-dev
apt-get install qtcreator qtbase5-private-dev qt5ct qtdeclarative5-dev qtdeclarative5-dev-tools
```

或者, 下载Qt5-Linux.run安装：

export Qt5_DIR=/opt/Qt5.12.12/5.12.12/gcc_64/lib/cmake/Qt5  # 注意路径

## 配置Qt5的环境变量
```
export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:/opt/Qt5.12.12/5.12.12/gcc_64/lib/cmake/
export PATH=/opt/Qt5.12.12/5.12.12/gcc_64/bin/:${PATH}
export LD_LIBRARY_PATH=/opt/Qt5.12.12/5.12.12/gcc_64/lib/:${LD_LIBRARY_PATH}
```

如果需要Paraview输出avi动画文件，还需要安装Ffmpeg库，源码下载：

如果在不支持OpenGL的电脑上运行paraview，还要安装MESA 3D库--模拟硬件的软件库。

# 安装VR插件
CMAKE编译参数添加：

-DBUILD_SHARED_LIB -DPARAVIEW_BUILD_QT_GUI -DPARAVIEW_USE_MPI  -DPARAVIEW_BUILD_PLUGIN_VRPlugin

需要安装OpenVR：
``` 
-- Installing: /usr/local/lib/libopenvr_api.a
-- Installing: /usr/local/include/openvr/openvr_driver.h
-- Installing: /usr/local/include/openvr/openvr_capi.h
-- Installing: /usr/local/include/openvr/openvr.h
-- Installing: /usr/local/share/pkgconfig/openvr.pc
```

## 使用VRUI库 
-DPARAVIEW_USE_VRUI   