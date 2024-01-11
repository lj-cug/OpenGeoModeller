# Ubuntu OS下安装不同版本R及在RStudio中设置

## 安装R-3.5.3
```
wget https://cran.rstudio.com/src/base/R-3/R-3.5.3.tar.gz
./configure --prefix=/opt/R/3.3.0 --enable-R-shlib --with-blas --with-lapack
make
make install
```

## 在RStudio中设置不同版本的R

which R

apt install r-base 在Ubuntu-20.04中安装的是R-4.2, 默认安装路径是：
/usr/bin/R

如果which R不能定位R, 则扫描/usr/local/bin 和 /usr/bin 路径下的R脚本

如果你要设置需要版本的R, 可以设置:
export RSTUDIO_WHICH_R=/usr/local/bin/R

环境变量设置要在 ~/.profile 文件中