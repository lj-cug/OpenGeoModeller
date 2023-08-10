# SCHISM模型中的NETCDF_SINGLE_OUTPUT

NETCDF_SINGLE_OUTPUT函数作用：在SCHISM模型中，并行计算时间迭代步中，自动合并为一个文件，输出为nc格式。

输出结果：outputs/schism_history.nc

## 结构：

[schism_steps.f90:]{.mark}

do it=1,nsteps

call NETCDF_SINGLE_OUTPUT(it)

enddo

SUBROUTINE NETCDF_SINGLE_OUTPUT(it)

IMPLICIT NONE

integer, intent(in) :: it

recs_his = recs_his + 1

Print \*, \' recs_his=\', recs_his

CALL WRITE_SINGLE_OUTPUT_DATA(it)

END SUBROUTINE

SUBROUTINE WRITE_SINGLE_OUTPUT_DATA(it)

iret=nf90_open(TRIM(FILE_NAME), nf90_write, ncid)

CALL GENERIC_NETCDF_ERROR_SCHISM(CallFct, 1, iret)

eTimeDay = eTimeStart + it \* (dt/86400.)

CALL WRITE_NETCDF_TIME_SCHISM(ncid, recs_his, eTimeDay)

! 一维整数型数组输出

CALL WRITE_1DVAR_SINGLE_INT(ncid, \"kbp00\", kbp00)

! 一维浮点数数组输出

CALL WRITE_1DVAR_SINGLE(ncid, \"eta2\", eta2)

! 标量输出

CALL WRITE_NVRT_KBP_SINGLE(ncid, \"qnon\", qnon)

！目前，还没有其他标量（泥沙，水质）的整体nc文件输出

! 考虑数据量很大时，单个文件输出速度很慢！

END

SUBROUTINE WRITE_1DVAR_SINGLE(ncid, string, VARin)

CALL GET_NETCDF_VARNAME(string, stringCF)

IF (myrank .eq. 0) THEN

allocate(VARout(np_global), stat=istat)

DO iProc=2,nproc

call
mpi_irecv(VARout,1,netcdf_his1_type(iProc-1),iProc-1,8024,comm,netcdf_his1_rqst(iProc-1),ierr)

END DO

DO IP=1,npa

IPglob = iplg(IP)

VARout(IPglob) = VARin(IP)

END DO

iret=nf90_inq_varid(ncid, TRIM(stringCF), var_id)

CALL GENERIC_NETCDF_ERROR_SCHISM(CallFct, 1, iret)

IF (nproc \> 1) THEN

call mpi_waitall(nproc-1,netcdf_his1_rqst,netcdf_his1_stat,ierr)

END IF

IF (NF90_RUNTYPE == NF90_OUTTYPE_HIS) THEN

iret=nf90_put_var(ncid,var_id,VARout,start = (/1, recs_his/), count = (/
np_global, 1 /))

CALL GENERIC_NETCDF_ERROR_SCHISM(CallFct, 2, iret)

ELSE

iret=nf90_put_var(ncid,var_id,SNGL(VARout),start = (/1, recs_his/),
count = (/ np_global, 1 /))

CALL GENERIC_NETCDF_ERROR_SCHISM(CallFct, 3, iret)

ENDIF

deallocate(VARout)

ELSE ! mrank /= 0

CALL MPI_SEND(VARin, npa, rtype, 0, 8024, comm, ierr)

END IF

END
