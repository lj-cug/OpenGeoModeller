# glmgui-install

## 安装依赖库

devtools::install_github("jsta/glmgui")

一些库没有在CRAN上, 需要单独从github安装(注意安装依赖顺序)：
```
devtools::install_github("GLEON/GLMr")
devtools::install_github("GLEON/GLM3r")
devtools::install_github("GLEON/glmtools")

devtools::install_github("jverzani/gWidgets2RGtk2")
devtools::install_github("lawremi/RGtk2")
```

## libgtk2
```
apt-get install gnome-core-devel 
apt-get install pkg-config
apt-get install libglib2.0-doc libgtk2.0-doc
apt-get install glade-gnome glade-common glade-doc 
apt-get install glade libglade2-dev
apt-get install libgtk2.0-dev
```

### 使用外部命令查看安装的gtk库版
```
pkg-config --modversion gtk+-2.0
pkg-config --version (查看pkg-config的版本)
pkg-config --list-all grep gtk (查看是否安装了gtk)
```

### gtk程序编译
```
gcc test.c `pkg-config --cflags --libs gtk+-2.0`
pkg-config --cflags --libs gtk+-2.0
gcc test.c `pkg-config --cflags --libs gtk+-2.0`, pkg-config --cflags --libs gtk+-2.0
```

### rGTK2问题
```
https://hub.fgit.cf/lawremi/RGtk2/issues/8

To build the extant version of RGtk2 on Windows, use R 4.1.3. Or you can get the binaries with install.packages("https://access.togaware.com/RGtk2_2.20.36.2.zip", repos=NULL) for the same version of R. It can't be done with the current version of R (4.2).

After installing it, run library(RGtk2); you'll be prompted to install Gtk+.
```

### gWidgets2RGtk2问题
```
https://hub.fgit.cf/jverzani/gWidgets2RGtk2/issues/31

it worked with install_github("jverzani/gWidgets2RGtk2"), but I had use R 3.6.3 (on windows).
```
