# WRF_v4.2-Run

wget -c -t 10 https://www2.mmm.ucar.edu/wrf/src/conus12km.tar.gz

tar xf conus12km.tar.gz --strip 1 -C /usr/local/wrf/conus12km

cp -r /usr/local/wrf/wrf-4.2/run/* conus12km

cd conus12km

ln -sf /usr/local/wrf/wrf-4.2/main/*.exe

���ڵ㣺
mpirun --allow-run-as-root -np 64 ./wrf.exe

��Ⱥ��
mpirun --allow-run-as-root -hostfile hostfile -x PATH=$PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH -np 256 �CN 128 ./wrf.exe

## �鿴

tail rsl.out.0000

wrf: SUCCESS COMPLETE WRF, ��ʾ�ɹ�!


