# install ARWPost
## ����Դ��
wget https://www2.mmm.ucar.edu/wrf/src/ARWpost_V3.tar.gz

## ��ѹARWpost
```
tar -zxvf ARWpost_V3.tar.gz
cd ARWpost
```
## ��װARWpost
./configure

����ѡ��gfortran��ѡ��3

����ARWpos�Ļ�����װ�������

## �޸�Makefile�ļ�
```
λ�ã�\Build_WRF\ARWpost\src\Makefile
����linux������������ֱ��ȥ�ļ��������޸ľ����ˣ��Ҹ����±���
��19�У�-L$(NETCDF)/lib -I$(NETCDF)/include -lnetcdf
�޸�Ϊ��-L$(NETCDF)/lib -I$(NETCDF)/include -lnetcdff -lnetcdf
���沢�˳�
```

## �޸�configure.arwp�ļ�
```
λ�ã�\Build_WRF\ARWpost\configure.arwp
����linux������������ֱ��ȥ�ļ��������޸ľ����ˣ��Ҹ����±���
��37�У�CFLAGS = -m64
�޸�Ϊ��CFLAGS = -fPIC -m64

��38�У�CPP = /lib/cpp -C -P -traditional
�޸�Ϊ��CPP = /lib/cpp -P -traditional

���沢�˳�
```

## ���ɿ�ִ���ļ�EXE
```
�ļ��޸Ľ����Ժ���ARWpost�ļ�����
���룺
./compile

ls -ls *.exe
���ֿ�ִ���ļ�ARWpost.exe
������ARWpost��������װ�����
```

## �޸�namelist.ARWpost
λ�ã�\Build_WRF\ARWpost\namelist.ARWpost

���ļ��Ժ󣬸����������ݽ���ȫ���ǣ�
```
&datetime
 start_date = '2021-10-15_00:00:00',
 end_date   = '2021-10-17_18:00:00',
 interval_seconds = 3600,
 tacc = 0,
 debug_level = 0,
/

&io
 input_root_name = '/home/�������������Լ����ļ�����/Build_WRF/WRF/test/em_real/wrfout_d01_2021-10-15_00:00:00'
 output_root_name = '/home/�������������Լ����ļ�����/Build_WRF/test_20211015'
 plot = 'all_list'
 fields = 'height,geopt,theta,tc,tk,td,td2,rh,rh2,umet,vmet,pressure,u10m,v10m,wdir,wspd,wd10,ws10,slp,mcape,mcin,lcl,lfc,cape,cin,dbz,max_dbz,clfr'
 output_type = 'grads' 
 mercator_defs = .true.
/
 split_output = .true.
 frames_per_outfile = 2


 plot = 'all'
 plot = 'list' 
 plot = 'all_list'
! Below is a list of all available diagnostics
 fields = 'height,geopt,theta,tc,tk,td,td2,rh,rh2,umet,vmet,pressure,u10m,v10m,wdir,wspd,wd10,ws10,slp,mcape,mcin,lcl,lfc,cape,cin,dbz,max_dbz,clfr'
 

&interp
 interp_method = 1,
 interp_levels = 1000.,975.,950.,925.,900.,875.,850.,825.,800.,750.,700.,650.,600.,550.,500.,450.,400.,350.,300.,250.,200.,150.,100.,
/
extrapolate = .true.

 interp_method = 0,     ! 0 is model levels, -1 is nice height levels, 1 is user specified pressure/height

 interp_levels = 1000.,975.,950.,925.,900.,875.,850.,825.,800.,750.,700.,650.,600.,550.,500.,450.,400.,350.,300.,250.,200.,150.,100.,
 interp_levels = 0.25, 0.50, 0.75, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
```

������������Ҫ�޸��ļ������ĵط��Ա�ע.