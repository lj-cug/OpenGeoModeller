# pytecplot学习

## python and Tecplot

需要安装tecplot360 2017 R1或更高版本的软件

python 64位 2.7 或 3.4+

## pytecplot安装

easy_install pip

pip install pytecplot或者python -m pip install pytecplot

本地源程序安装：python -m pip install .

管理员权限：python -m pip install \--user pytecplot

可选安装其他库：python -m pip install pytecplot\[extras\]

extras：Numpy IPython

环境变量设置：

\"C:\\Program Files\\Tecplot\\Tecplot 360 EX \[VERSION\]\\bin\"

echo %PATH%

更新软件：python -m pip install \--upgrade pytecplot

## 与Tecplot 360 GUI连接

需要2017 R3或更高版本，TecUtil Server addon

Scripting -\> PyTecplot Connections...，默认端口是7600

（1）从客户端连接

import tecplot

tecplot.session.connect(port=7600)

（2）使用宏文件激活服务器

.mcr文件：

\$!EXTENDEDCOMMAND

COMMANDPROCESSORID = \"TecUtilServer\"

COMMAND = R\"(

AcceptRequests = Yes

ListenOnAddress = localhost

ListenOnPort = 7600

)\"

然后：tec360 -p startConnector.mcr

# 使用handy python scripts

## 安装pytecplot

pip install pytecplot

## 提取垂向剖面

（1）Define the Vertical Transect Path

使用polyline geometry tool，定义要提取的剖面多段线

（2）提取点

鼠标右键，选择extract

（3）启动PyTecplot

PyTecplot Connections via Scripting -\> PyTecplot Connections

python -O VerticalTransect.py  (需要安装numpy: pip install numpy)

安装pip: easy_install.exe pip

（4）Animate Through Time

## 图片转PLT

安装pip install Pillow

图片分辨率高时，计算量和形成的plt文件都很大。

## shapefile转plt

比如WRF模拟的气旋移动，可以在shapefile上显示，明确定位。

添加text到图中时：例如，transect@&(SOLUTIONTIME)
