# glmgui-install

## ��װ������

devtools::install_github("jsta/glmgui")

һЩ��û����CRAN��, ��Ҫ������github��װ(ע�ⰲװ����˳��)��
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

### ʹ���ⲿ����鿴��װ��gtk���
```
pkg-config --modversion gtk+-2.0
pkg-config --version (�鿴pkg-config�İ汾)
pkg-config --list-all grep gtk (�鿴�Ƿ�װ��gtk)
```

### gtk�������
```
gcc test.c `pkg-config --cflags --libs gtk+-2.0`
pkg-config --cflags --libs gtk+-2.0
gcc test.c `pkg-config --cflags --libs gtk+-2.0`, pkg-config --cflags --libs gtk+-2.0
```

### rGTK2����

ref: rGTK2-install.md
