# R-4.3.2��װ��curl�Ĵ���

https://github.com/jeroen/curl/issues/310

You have multiple conflicting versions of libcurl on your machine. Try removing the custom one under /usr/local:

sudo rm -Rf /usr/local/include/curl 

sudo rm -Rf /usr/local/lib/libcurl*

And instead install the libcurl system library on your machine, such as yum install curl-devel or apt-get install libcurl4-openssl-dev

# ��װR-3.5.3��devtools��װ����
```
install.packages("devtools")
install.packages("ragg")
install.packages("textshaping")
```
��װ������������

## ���
```
sudo apt-get install autoconf automake libtool
sudo apt-get install libgtk-3-dev
```
