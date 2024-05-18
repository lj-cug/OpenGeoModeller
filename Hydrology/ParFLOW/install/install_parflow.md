# build parflow
https://www.jianshu.com/p/5be68f4c65ca

## �����Լ������Դ�ο���ַ
```
Github:https://github.com/parflow/parflow
Parflow blog:http://parflow.blogspot.com/
Parflow users:https://mailman.mines.edu/pipermail/parflow-users/
User manual��https://github.com/parflow/parflow/blob/master/user_manual.pdf
```

## ����parflow
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

## ����example����parflow
```
mkdir foo
cd foo
cp $PARFLOW_DIR/examples/default_single.tcl .   (ע������и��� .)
chmod 640 * (foo�����ļ���дȨ�ޣ�
tclsh default_single.tcl
```

ͬʱ��foo�ļ����»�����ģ�����еĽ��.

## ���ܳ��ֵĴ���
1 build �� install Parflow�����г��֣�����ͼ��ʾ����

����취����pftools�ļ����µ�pkgindex.tcl�ļ����������滻Ϊ��

Ҳ��������ȥgithub���ظ��ļ�?https://github.com/parflow/parflow/blob/master/pftools/pkgIndex.tcl

2 build ·���£�ִ��make install ����� ���� /config permission denied ����

����취��������ȷ��PARFLOW_DIR·�����ڱ���������Ϊ /home/pf/pfdir/parflow�� ��ͨ��ccmake���CMAKE_INSTALL_PREFIX�Ƿ���PARFLOW_DIRһ��

���õ�ǰ����·����build��, ���д���Ϊ:

ccmake ../parflow

��������ͼ��ʾ��GUI.

������/home/pf/pfdir/parflow���밴������������ʾ���޸�·����
�޸ĺ�, �Ȱ� c ����configure, Ȼ�����µ���ʾ���水 e, ��� g �����˳�GUI
