# ����SCHISM_v5.9.0�����ϰ汾

## ����Դ��

git clone https://github.com/schism-dev/schism.git

cd schism

git checkout tags/v5.10.0  #- for v5.10.0.

## ����

��ҪFORTRAN��C������(MPI wrapper, ��mpif90, mpicc), NetCDF (version 4.4 and above), Python (version 2.7 and above)

### GNU Make

cp Make.defs.bora Make.defs.myown

ln -sf Make.defs.myown Make.defs.local

�༭Make.defs.local����MPI���������ƣ�NetCDF4��·��(-lnetcdff for FORTRAN)�Ϳ�ִ�г�������(EXEC)

��/�ر�**include_modules**�е����ģ��(TVD_LIM������ֵ)

ȷ��mk/sfmakedepend.pl��mk/cull_depends.pl�п�ִ��Ȩ�޵�(chmod+x)

cd ../src

make clean

make pschism

ע�⣺�������git clone��Դ�룬����ʱ�����

������ɿ�ִ�г���pschism_* (*����򿪵����ģ�����ƣ���SED etc.)

### CMAKE

��Ҫ2���ļ���

1. SCHISM.local.build: ��/�ر�ѡ��ģ��(��include_modules����). 

TVD_LIM��������ֵ, �������ģ��رգ������ˮ����ģ��. 

NO_PARMETIS: �ƹ�ParMETIS��, ����ʱ��Ҫ�ṩһ������ֽ�ͼpartition.prop (��ParMETIS�����global_to_local.propһ��)

OLDIO: ����ȫ������Ŀ���. ʵʩ�첽I/O (aka 'scribed' I/O)���ϲ�ȫ�ֱ�������scribes����������ļ���ʹ�ø�ѡ���Ҫ�ر�OLDIO���û���Ҫ��������ָ��scribe���ı�ţ���ϸ��Ϣ�ο�Run-the-model.md�������OLDIO����ʹ��֮ǰ��I/Oģʽ(ÿ��MPI���̶����)���û���Ҫʹ�ú���ű��ϲ����.

2. SCHISM.local.cluster_name: ��Make.defs.local���ƣ����ļ���������Ҫ�Ļ������������������NetCDF������ƺ�·��

cp -L SCHISM.local.whirlwind SCHISM.local.myown  # ʹ�����е�SCHISM.local.cluster_name

mkdir ../build

cd ../build; rm -rf * # Clean old cache

cmake -C ../cmake/SCHISM.local.build -C ../cmake/SCHISM.local.myown ../src/

���CMAKE���ú�ִ�У�

make -j8 pschism

����

make VERBOSE=1 pschism # serial build with a lot of messages

������ɵĿ�ִ�г���pschism_*��build/bin/���������build/lib/�����߽ű���bin/
