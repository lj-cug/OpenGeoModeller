# 编译RegCM4
```
cp RegCM-4.5.0.tar.gz ./
tar xzvf RegCM.tar.gz
cd RegCM-4.5.0
./configure CC=icc FC=ifort MPIFC=mpiifort --enable-clm
make
make install
```

# 设置RegCM4
在RegCM-4.5.0同一级目录下：
```
mkdir Run_RegCM
cd Run_RegCM
mkdir input output
ln -sf ../RegCM-4.5.0/bin ./
cp ../RegCM-4.5.0/Testing/test_002.in ./
vi test_002.in 修改所需数据路径，修改区域时间
vi ../RegCM-4.5.0/Doc/README.namelist 可以查看内容
```

# 运行RegCM4
```
./bin/terrainCLM test_002.in # 地形
./bin/sstCLM test_002.in     # CLM-海表温度
./bin/icbcCLM test_002.in    # CLM初始和边界场

mpirun -np 32 ./bin/regcmMPICLM test_002.in
```