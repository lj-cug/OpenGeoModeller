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

wget https://curl.se/download/curl-7.85.0.tar.gz

tar -vxzf curl-7.85.0.tar.gz

cd curl-7.85.0

./configure --with-openssl

make

make install