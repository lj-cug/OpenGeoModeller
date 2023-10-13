# 解决R安装curl的错误

https://github.com/jeroen/curl/issues/310

You have multiple conflicting versions of libcurl on your machine. Try removing the custom one under /usr/local:

sudo rm -Rf /usr/local/include/curl 

sudo rm -Rf /usr/local/lib/libcurl*

And instead install the libcurl system library on your machine, such as yum install curl-devel or apt-get install libcurl4-openssl-dev