# ����RegCM4
```
cp RegCM-4.5.0.tar.gz ./
tar xzvf RegCM.tar.gz
cd RegCM-4.5.0
./configure CC=icc FC=ifort MPIFC=mpiifort --enable-clm
make
make install
```

# ����RegCM4
��RegCM-4.5.0ͬһ��Ŀ¼�£�
```
mkdir Run_RegCM
cd Run_RegCM
mkdir input output
ln -sf ../RegCM-4.5.0/bin ./
cp ../RegCM-4.5.0/Testing/test_002.in ./
vi test_002.in �޸���������·�����޸�����ʱ��
vi ../RegCM-4.5.0/Doc/README.namelist ���Բ鿴����
```

# ����RegCM4
```
./bin/terrainCLM test_002.in # ����
./bin/sstCLM test_002.in     # CLM-�����¶�
./bin/icbcCLM test_002.in    # CLM��ʼ�ͱ߽糡

mpirun -np 32 ./bin/regcmMPICLM test_002.in
```