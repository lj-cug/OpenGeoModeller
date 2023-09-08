# Install telemac-v8

�ο���  http://wiki.opentelemac.org/doku.php?id=installation_on_linux

## ��Ҫ�Ĺ���

Python 3.7+
Numpy 1.15+
A Fortran compiler (GFortran 4.6.3 is the minimum)
MPI
METIS-5.1
SCOTCH
ParMETIS
PT-SCOTCH
SciPy
Matplotlib
GOTM

## ����Դ��

wget https://gitlab.pam-retd.fr/otm/telemac-mascaret/-/archive/v8p3r1/telemac-mascaret-v8p3r1.tar.gz

����

git clone https://gitlab.pam-retd.fr/otm/telemac-mascaret.git my_opentelemac

cd my_opentelemac

git checkout tags/v8p4r0

## ��װǰ��˵��

<root> ��TELEMAC-MASCARET��Դ��·��

<systel.cfg> �ǹ��������ļ�

<config> ָ��ʹ�õĹ��������ļ�

<pysource> �Ǳ��뻷���ļ�

## ���ñ��뻷��

������δ���<pysource>�ļ���������ҵ�һ��ģ���ļ��� <root>/configs/pysource.template.sh

�������»���������

export HOMETEL=$HOME/telemac-mascaret
export SYSTELCFG=$HOMETEL/configs/systel.cfg
export USETELCFG=gfortranHPC
export SOURCEFILE=$HOMETEL/configs/pysource.gfortranHPC.sh

export METISHOME=~/opt/metis-5.1.0

�ο� pysource.sh

������~/.bashrc��������£�

source $HOME/telemac-mascaret/configs/pysource.gfortranHPC.sh

### ���ػ���

source pysource.gfortranHPC.sh

## ����TELEMAC-MASCARET

������δ���<systel.cfg>�ļ�����<root>/configs����һЩ�����ļ�ģ��

�ο� systel.cfg

## ����TELEMAC-MASCARET

source pysource.gfortranHPC.sh

��ʾ���ã�  config.py

��ʼ��������ϵͳ�� compile_telemac.py

## ������ҵ

cd  <root>examples/telemac2d/gouttedo

telemac2d.py t2d_gouttedo.cas --ncsize=4





