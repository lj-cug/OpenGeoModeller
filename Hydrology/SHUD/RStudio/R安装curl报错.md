# 解决R安装curl的错误

参考： https://blog.csdn.net/qq_34536369/article/details/131579825

## 安装 openssl

apt install libcurl4-openssl-dev

## 手动指定INCLUDE_DIR和LIB_DIR

先得找到libcurl.pc的位置

可以直接sudo find / -name "libcurl.pc"，查的有点慢

直接找默认应该有在/usr/lib/x86_64-linux-gnu/pkgconfig/libcurl.pc

## 配置Renviron

命令gedit ~/.Renviron

我是个空文件，写入：

INCLUDE_DIR=/usr/lib/x86_64-linux-gnu/pkgconfig/libcurl.pc
LIB_DIR=/usr/lib/x86_64-linux-gnu/pkgconfig/libcurl.pc

然后再打开R去装curl就成功了
