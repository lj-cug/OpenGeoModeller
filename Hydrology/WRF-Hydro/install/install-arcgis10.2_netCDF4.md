# 设置python2.7的环境变量
set PATH=D:\Program Files\ArcGIS\Python27\ArcGIS10.2;%PATH%

因为ArcGIS 10.2使用低版本的Python2.7, 因此都要手动安装2013年附近的低版本的相关Python库。
目的是使用import netCDF4

# install pip

从 pypi下载pip-1.5

cd pip-1.5

python setup.py install

## 设置pip2的环境变量
set PATH=D:\Program Files\ArcGIS\Python27\ArcGIS10.2\Scripts;%PATH%

# 安装numpy-1.8.0
python setup.py install

安装过程中使用Visual Studio 2008编译源码

# 安装netCDF4-1.0.6.win32-py2.7.exe
不能使用X64的安装程序, 只能使用win32

注册python2: python registry.py

然后安装netCDF4-1.0.6.win32-py2.7.exe

# 测试python2.7-netCDF4
python

>>import netCDF4


