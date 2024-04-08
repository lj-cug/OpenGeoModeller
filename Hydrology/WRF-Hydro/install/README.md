# 编译WRF-Hydro

如果使用版本过低的NetCDF-Fortran，则出现错误：
2个NC API的问题： nf90_def_var_deflate  和 nf_get_var_int64

NETCDF-FORTRAN必须使用>4.5.1
