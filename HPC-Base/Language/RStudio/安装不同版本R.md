# Ubuntu 20.04�°�װ��ͬ�汾R����RStudio������

## ��װR-3.5.3
```
#�ο�: https://stackoverflow.com/questions/20752307/error-in-install-previous-versions-of-r-on-ubuntu
apt-get install xorg-dev libbz2-dev libcurl4-openssl-dev
conda deactivate  // ������conda������, ����libcurl�汾������

wget https://cran.rstudio.com/src/base/R-3/R-3.5.3.tar.gz
tar xvf R-3.5.3.tar.gz && cd R-3.5.3
./configure --prefix=/opt/R/3.5.3 --enable-R-shlib --with-blas --with-lapack
make
make install
```

## ��RStudio�����ò�ͬ�汾��R

which R

apt install r-base ��Ubuntu-20.04�а�װ����R-4.2, Ĭ�ϰ�װ·���ǣ�
/usr/bin/R

���which R���ܶ�λR, ��ɨ��/usr/local/bin �� /usr/bin ·���µ�R�ű�

�����Ҫ������Ҫ�汾��R, ��������:
export RSTUDIO_WHICH_R=/opt/R/3.5.3

������������Ҫ�� ~/.profile �ļ���