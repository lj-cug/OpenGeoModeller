# singularity�İ�װ��ʹ��

## ��װ����

sudo apt-get update && sudo apt-get install -y 
build-essential 
uuid-dev  
libgpgme-dev 
squashfs-tools  
libseccomp-dev  
cryptsetup-bin

## ��װGO����

���ص�ַ��https://golang.org/dl/

wget https://studygolang.com/dl/golang/go1.21.1.src.tar.gz     #����

tar -C /pub/software -xzvf go1.21.1.src.tar.gz                 #��ѹ

rm go1.21.1.src.tar.gz                                #ɾ����װ��

### ��ӵ���������

echo 'export PATH=/pub/software/go/bin(���·��):$PATH' >> ~/.bashrc

## ����singularity

���ص�ַ��https://github.com/hpcng/singularity/releases

wget https://github.com/hpcng/singularity/releases/download/v3.7.2/singularity-3.7.2.tar.gz

tar -xzf singularity-3.7.2.tar.gz #��ѹ

cd singularity

## ��װsingularity

./mconfig

cd builddir

make

sudo make install

�ǵ���ӵ���������

# ��������

## ���ؾ���

���Դ� Container Library��https://cloud.sylabs.io/library��

���磺

singularity pull library://cenat/default/blast.sif:latest

or Docker Hub��https://hub.docker.com/)����images��

���磺

singularity pull docker://ncbi/blast

singularity pull --arch amd64 library://library/default/ubuntu:20.04

### ��������

Ĭ�ϻ��Զ�����home/PWD , /tmp , /proc , /sys , /dev Ŀ¼��

singularity shell --writable --fakeroot blast

�������а�װ��������鲻Ҫʹ��anaconda ��װ�������ֶ���װ������Ҫ������������������

��ӻ�������

�˳�������, ��blast/environment �����PATH

"
vi blast/environment
!/bin/sh
export PATH=/opt/ncbi-blast-2.10.1+/bin:$PATH
"

### ���

���ȫ����װ���֮���������

singularity build blast.sif blast

### ���г���

singularity exec blast.sif  blasp XXX �����������÷�

## ��������

### ����ʽ����

singularity shell blast.sif bash

### ֱ������

singularity exec blast.sif blastp

### �û���Ȩ��

ʹ���������ò����ǰ�ȫ�ԣ���ȫ�������������棬һ����ʹ���˲������ε�����������������ڵ����ϰ�װ�˲������ε����һ����Singularity�ṩ��ǩ����������֤��
��һ�����������᲻��ԽȨ��Host��һЩ�����������飬�������Ҫ���ǵġ�

singularity �Ľ���취�ǻ��������ڶ�̬����һ���û������û���Host����û�����������Ȩ�޵ȶ�����һ�¡�
��������Host �������˵����飬��������Ҳ�ɲ��ˡ�