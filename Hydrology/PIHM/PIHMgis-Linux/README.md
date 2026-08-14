# PIHMgis-Linux

## 修改Makefile

修改Makefile中链接Qt的路径

## Ubuntu 20.04 安装QT5

```bash
sudo apt-get install build-essential
sudo apt-get install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools
sudo apt-get install qt5*
```

## 启动PIHMgis

需要libgdal_1.so，执行：

```bash
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
```

## 运行

```bash
./PIHMgis
```

## 相关文档

- [PIHM](../)：模型入口
- [Hydrology](../../)：学科入口
- [PIHM/install](../install/)：MM-PIHM 安装
- [PIHM/doc](../doc/)：原理与输入说明
