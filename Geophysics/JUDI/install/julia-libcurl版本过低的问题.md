# julia-curl版本过低的问题

## 查看

查看julia目前使用的libcurl

using Libdl

filter!(contains("curl"), dllist())

## libcurl版本过低的问题

会引起加载HDF5库的问题：using HDF5;

# 安装新版本的libcurl

apt install libcurl4-openssl-dev libcurl4-doc libidn11-dev libkrb5-dev libldap2-dev librtmp-dev libssh2-1-dev libssl-dev zlib1g-dev

或者源码安装：

wget https://curl.se/download/curl-7.85.0.tar.gz

tar -vxzf curl-7.85.0.tar.gz

cd curl-7.85.0

./configure --with-openssl

make

make install