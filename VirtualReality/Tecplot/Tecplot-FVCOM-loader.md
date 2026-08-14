# **FVCOM loader (Tecplot 2018 R2)**

使用FVCOM加载工具可以导入FVCOM模型输出的netCDF格式文件到Tecplot 360
EX。目前，支持经典的netCDF和netCDF-4格式。

（1）一个或多个具有相同的拓补结构和变量结构FVCOM输出结果，可以使用该加载工具导入到一个单独的Tecplot数据集，每个时间步创建一个zone。

（2）FVCOM历史输出文件具有不同的拓补结构和变量结构，只能通过appending的方式进行合并。

X和Y变量直接从文件加载，然后投影到各Z分层位置，Z分层位置由zeta, h,
siglev变量计算得到。

基于FVCOM属性数据中定义的坐标系统来选择网格变量。如果没有提供坐标系统，假设为Cartesian坐标。

加载工具将添加下列附属数据到数据集中：

![](./media/image1.emf){width="5.249773622047244in"
height="1.3140868328958881in"}

添加垂向分层和分层变量来可视化FVCOM模型分层。各sigma分层的节点上的FVCOM变量加载为节点类型变量，各sigma分层的单元上的FVCOM变量加载为单元中心类型变量。

存储于各sigma分层的节点上的FVCOM变量插值到单元体中心位置，忽略任何FVCOM定义的边界条件。忽略存储于各sigma分层的单元上的变量。

文件属性和各变量分别加载到数据集和变量属性数据。

# FVCOM_loader测试

测试FVCOM3.1模型计算lock exchange算例的netCDF输出文件tst.nc，导入成功。

# FVCOM2.7_netCDF_IO

FVCOM_v2.7代码中：

[program main]{.mark}

主程序us_fvcom.f90输出计算结果，调用archive子程序，其中输出nc文件

**DO** IINT=ISTART,IEND

。。。

call ARCHIVE

END DO

END

[SUBROUTINE ARCHIVE]{.mark}

\# if defined (NETCDF_IO)

USE MOD_NCDIO ! 各时刻计算结果

USE MOD_NCDAVE ! 时间平均计算结果

\# endif

！ 输出nc文件

\# if defined (NETCDF_IO)

！按一定频率输出

IF(MOD(IINT,CDF_INT)==0) CALL OUT_NETCDF

\# endif

！ 还有其他结果输出，如时间平均值、泥沙、冰、balance_2D等

END SUBROUTINE ARCHIVE

! Write Time Dependent NetCDF Data to File

subroutine out_netcdf

use all_vars

use netcdf

dims(1) = stck_cnt

!\--Write Header information if first output of file

call write_netcdf_setup(1)

!\--Open File

ierr = nf90_open(cdfname,nf90_write,nc_ofid)

end

# combine_output8

SCHISM模型的combine_output8程序合并rank-specific的二进制格式文件，然后输出nc文件：

！进入定义nc模式

iret = nf_create(trim(fname), NF_CLOBBER, ncid)

！定义维度

iret = nf_def_dim(ncid, \'nele\',ne_global, nele_dim)

！以下就是定义变量和输出变量属性的描述文字

iret=nf_def_var(ncid,\'time\',NF_REAL,1,time_dims,itime_id) ! 定义变量

iret=nf_put_att_text( ) ! 输出该变量的属性描述文字

！离开nc定义模式

iret = nf_enddef(ncid)

！进入输出模式 (header part only)

iret=nf_put_vara_int( )

iret=nf_put_vara_real( )

!end output header

! Loop over output spools in file

**do** ispool=1,nrec

!Gather all ranks

**do** irank=0,nproc-1

!Open input file

！输出nc格式的变量内容，跟上面的一样。

iret=nf_put_vara_real()

**enddo** !ispool=1,nrec

! 关闭nc输出文件

iret = nf_close(ncid)

**enddo** ! do iinput=ibgn,iend
