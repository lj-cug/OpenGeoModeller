# ����SCHISM_v5.3.1�Ĳ���

��Ҫmk/·������Make.defs.myown

cp Make.defs.tsunami Make.defs.myown
ln -sf Make.defs.myown Make.defs.local

�༭ Make.defs.local������MPI��������netcdf��·���������netcdf 3.x)

ParMETIS��SHICSM����һ����룬��Ҫ����src/ParMetis-3.1-Sep2010/Makefile.in��MPI C���������ƣ��ο�ParMETIS·���µ�INSTALL����

Ȼ�󣬴򿪻�ر�include_modules�еĿ��أ�TVD_LIM����Ҫ���á�

Ȼ��
cd ../src 
make clean 
make

������ɿ�ִ�г���pschism_*    (*����򿪵����ģ�����ƣ���SED etc.)


