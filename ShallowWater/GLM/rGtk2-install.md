# install glmgui in R-4.3.2
```
devtools::install_github("GLEON/glmtools")

devtools::install_github("jsta/glmgui")

错误：ERROR: dependencies ‘gWidgets2RGtk2’, ‘RGtk2’, ‘GLMr’ are not available for package ‘glmgui’

解决： devtools::install_github("GLEON/GLMr")

错误：
library("GLMr")
GLMr::glm_version()
/usr/local/lib/R/site-library/GLMr/exec/nixglm: error while loading shared libraries: libpng12.so.0: cannot open shared object file: No such file or directory

在R-4.3.2上安装GTK图形库：  RGtk2和gWidgets2RGtk2, 使用CRAN Archive上下载的源码, 本地安装：

install.packages("/home/lijian/Lake/GLM/RGtk2_2.20.36.3.tar.gz", repos=NULL, type="source")
错误1：
checking for GTK... no
configure: error: GTK version 2.8.0 required
解决：
apt install libgtk-2.0
修改RGtk2的代码： /src/RGtk2/RCommon.h

错误2：
Error in dyn.load(file, DLLpath = DLLpath, ...) : 
修改RGtk2的代码： /src/RGtk2/RCommon.h  // Added by LIjian

Failed to load module "canberra-gtk-module"
解决： apt-get install libcanberra-gtk-module

install.packages("/home/lijian/Lake/GLM/gWidgets2RGtk2_1.0-7.tar.gz", repos=NULL, type="source")

运行glmgui:
library("glmgui")
glmGUI()

退出路径, 会出错?!
```
