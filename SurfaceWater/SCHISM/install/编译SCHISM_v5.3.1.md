# 编译SCHISM_v5.3.1的步骤

需要mk/路径下有Make.defs.myown

cp Make.defs.tsunami Make.defs.myown
ln -sf Make.defs.myown Make.defs.local

编辑 Make.defs.local，定义MPI编译器，netcdf的路径（最好是netcdf 3.x)

ParMETIS与SHICSM代码一块编译，需要更新src/ParMetis-3.1-Sep2010/Makefile.in中MPI C编译器名称（参考ParMETIS路径下的INSTALL）。

然后，打开或关闭include_modules中的开关，TVD_LIM总是要设置。

然后，
cd ../src 
make clean 
make

最后，生成可执行程序：pschism_*    (*代表打开的相关模块名称，如SED etc.)


