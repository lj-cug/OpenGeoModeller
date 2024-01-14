# install GOTM-5.2.1
```
wget https://github.com/gotm-model/code/archive/v5.2.1.tar.gz
tar xzvf v5.2.1.tar.gz
cd code-5.2.1/src
wget http://basilisk.fr/src/gotm/gotm.patch?raw -O gotm.patch
patch -p0 < gotm.patch 
cd ..
mkdir build
cd build
cmake ../src -DGOTM_USE_FABM=off
make
```


