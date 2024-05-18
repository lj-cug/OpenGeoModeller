# build parflow
https://www.jianshu.com/p/5be68f4c65ca

## 下载以及相关资源参考地址
```
Github:https://github.com/parflow/parflow
Parflow blog:http://parflow.blogspot.com/
Parflow users:https://mailman.mines.edu/pipermail/parflow-users/
User manual：https://github.com/parflow/parflow/blob/master/user_manual.pdf
```

## 编译parflow
```
apt-get install gdc tcl-dev tk-dev
mkdir pfdir
cd pfdir
wget -c  #https://parflow.org/#download
tar -xvf parflow-3.3.1.tar.gz
cd parflow-3.3.1
mkdir build
cd build
export PARFLOW_DIR=/home/pf/pfdir/parflow
cd build
cmake ../parflow -DCMAKE_INSTALL_PREFIX=$PARFLOW_DIR 
make -j8
make install
```

## Test
```
cd $PARFLOW_DIR
cd test
tclsh default_single.tcl 1 1 1
```

## Run test
```
cd ../../build
make test
```

## 运行example测试parflow
```
mkdir foo
cd foo
cp $PARFLOW_DIR/examples/default_single.tcl .   (注意最后有个点 .)
chmod 640 * (foo赋予文件读写权限）
tclsh default_single.tcl
```

同时，foo文件夹下会生成模型运行的结果.

## 可能出现的错误
1 build 和 install Parflow过程中出现，如下图所示问题

解决办法：打开pftools文件夹下的pkgindex.tcl文件，将代码替换为：

也可以自行去github下载该文件?https://github.com/parflow/parflow/blob/master/pftools/pkgIndex.tcl

2 build 路径下，执行make install 后出现 如下 /config permission denied 错误

解决办法：设置正确的PARFLOW_DIR路径，在本次运行中为 /home/pf/pfdir/parflow， 且通过ccmake检查CMAKE_INSTALL_PREFIX是否与PARFLOW_DIR一致

设置当前工作路径在build下, 运行代码为:

ccmake ../parflow

出现如下图所示的GUI.

若不是/home/pf/pfdir/parflow，请按照最下栏的提示，修改路径。
修改后, 先按 c 进行configure, 然后在新的显示界面按 e, 最后按 g 保存退出GUI
