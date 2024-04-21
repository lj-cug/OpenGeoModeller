# install ARWPost
## 下载源码
wget https://www2.mmm.ucar.edu/wrf/src/ARWpost_V3.tar.gz

## 解压ARWpost
```
tar -zxvf ARWpost_V3.tar.gz
cd ARWpost
```
## 安装ARWpost
./configure

还是选择gfortran，选择3

这样ARWpos的基本安装就完成了

## 修改Makefile文件
```
位置：\Build_WRF\ARWpost\src\Makefile
建议linux不熟练的朋友直接去文件夹下面修改就行了，找个记事本打开
第19行：-L$(NETCDF)/lib -I$(NETCDF)/include -lnetcdf
修改为：-L$(NETCDF)/lib -I$(NETCDF)/include -lnetcdff -lnetcdf
保存并退出
```

## 修改configure.arwp文件
```
位置：\Build_WRF\ARWpost\configure.arwp
建议linux不熟练的朋友直接去文件夹下面修改就行了，找个记事本打开
第37行：CFLAGS = -m64
修改为：CFLAGS = -fPIC -m64

第38行：CPP = /lib/cpp -C -P -traditional
修改为：CPP = /lib/cpp -P -traditional

保存并退出
```

## 生成可执行文件EXE
```
文件修改结束以后，在ARWpost文件夹下
输入：
./compile

ls -ls *.exe
发现可执行文件ARWpost.exe
这样，ARWpost就真正安装完成了
```

## 修改namelist.ARWpost
位置：\Build_WRF\ARWpost\namelist.ARWpost

打开文件以后，复制以下内容进行全覆盖：
```
&datetime
 start_date = '2021-10-15_00:00:00',
 end_date   = '2021-10-17_18:00:00',
 interval_seconds = 3600,
 tacc = 0,
 debug_level = 0,
/

&io
 input_root_name = '/home/请在这里输入自己的文件夹名/Build_WRF/WRF/test/em_real/wrfout_d01_2021-10-15_00:00:00'
 output_root_name = '/home/请在这里输入自己的文件夹名/Build_WRF/test_20211015'
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

里面有两处需要修改文件夹名的地方以标注.