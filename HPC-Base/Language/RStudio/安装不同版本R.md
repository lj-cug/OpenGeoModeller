# Ubuntu OS�°�װ��ͬ�汾R����RStudio������

## ��װR-3.5.3
```
wget https://cran.rstudio.com/src/base/R-3/R-3.5.3.tar.gz
./configure --prefix=/opt/R/3.3.0 --enable-R-shlib --with-blas --with-lapack
make
make install
```

## ��RStudio�����ò�ͬ�汾��R

which R

apt install r-base ��Ubuntu-20.04�а�װ����R-4.2, Ĭ�ϰ�װ·���ǣ�
/usr/bin/R

���which R���ܶ�λR, ��ɨ��/usr/local/bin �� /usr/bin ·���µ�R�ű�

�����Ҫ������Ҫ�汾��R, ��������:
export RSTUDIO_WHICH_R=/usr/local/bin/R

������������Ҫ�� ~/.profile �ļ���