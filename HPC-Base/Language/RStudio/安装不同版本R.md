# Ubuntu 20.04下安装不同版本R及在RStudio中设置

## 安装R-3.5.3
```
#参考: https://stackoverflow.com/questions/20752307/error-in-install-previous-versions-of-r-on-ubuntu
apt-get install xorg-dev libbz2-dev libcurl4-openssl-dev
conda deactivate  // 不能在conda环境下, 否则libcurl版本不满足

wget https://cran.rstudio.com/src/base/R-3/R-3.5.3.tar.gz
tar xvf R-3.5.3.tar.gz && cd R-3.5.3
./configure --prefix=/opt/R/3.5.3 --enable-R-shlib --with-blas --with-lapack
make
make install
```

## 在RStudio中设置不同版本的R

which R

apt install r-base 在Ubuntu-20.04中安装的是R-4.2, 默认安装路径是：
/usr/bin/R

如果which R不能定位R, 则扫描/usr/local/bin 和 /usr/bin 路径下的R脚本

如果你要设置需要版本的R, 可以设置:
export RSTUDIO_WHICH_R=/opt/R/3.5.3

环境变量设置要在 ~/.profile 文件中