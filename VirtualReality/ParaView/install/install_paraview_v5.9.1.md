# Build ParaView-v5.9.1

## Download Code
```
git clone https://gitlab.kitware.com/paraview/paraview.git paraview

cd paraview
git checkout v6.1.1
git submodule update --init --recursive
```

或者，直接下载对应版本：
```
git clone -b v5.9.1 --recursive https://gitlab.kitware.com/paraview/paraview.git paraview-5.9.1
```

## 更新代码
```
git pull
git submodule update
```

## 初步设置CMAKE 
```
cd ../
mkdir build
cd build
cmake .. 
```

## 安装
```
make -j8
make install
```