# julia-curl�汾���͵�����

## �鿴

�鿴juliaĿǰʹ�õ�libcurl

using Libdl

filter!(contains("curl"), dllist())

## libcurl�汾���͵�����

���������HDF5������⣺using HDF5;

# ��װ�°汾��libcurl

apt install libcurl4-openssl-dev libcurl4-doc libidn11-dev libkrb5-dev libldap2-dev librtmp-dev libssh2-1-dev libssl-dev zlib1g-dev

����Դ�밲װ��

wget https://curl.se/download/curl-8.3.0.tar.gz    # 8.3.0 for julia-1.9.0

tar -vxzf curl-8.3.0.tar.gz

cd curl-8.3.0

./configure --with-openssl

make

make install

## ���û�������

gedit ~/.bashrc

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH