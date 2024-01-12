# R-4.3.2安装后curl的错误

https://github.com/jeroen/curl/issues/310

You have multiple conflicting versions of libcurl on your machine. Try removing the custom one under /usr/local:

sudo rm -Rf /usr/local/include/curl 

sudo rm -Rf /usr/local/lib/libcurl*

And instead install the libcurl system library on your machine, such as yum install curl-devel or apt-get install libcurl4-openssl-dev

# 安装R-3.5.3后devtools安装包括
```
install.packages("devtools")
install.packages("ragg")
install.packages("textshaping")
```
安装上述包都报错

## 解决
```
sudo apt-get install autoconf automake libtool
sudo apt-get install libgtk-3-dev
```
