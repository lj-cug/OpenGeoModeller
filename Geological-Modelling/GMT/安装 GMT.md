# Ubuntu/Debian �°�װ GMT

���� Ubuntu �� Debian �����������¹��򣬹ٷ����Դ�ṩ��ͨ�������ϰ汾 GMT������ GMT 5.4.5 ���� GMT 6.0.0�������� GMT �����°汾��

�����������������¼���ѡ��

ʹ���ϰ汾 GMT

ͨ�� conda ��װ GMT ���Ƽ���

Linux/macOS �±��� GMT Դ�� ���Ƽ���

ͨ������������԰�װ Ubuntu/Debian �ٷ�Դ�ṩ�� GMT �����ư���

����������б�:
$ sudo apt update

��װ GMT:
$ sudo apt install gmt gmt-dcw gmt-gshhg

��װ GMT ��ع���

���� PDF��JPG ��ͼƬ��ʽ��Ҫ Ghostscript�����룩:
$ sudo apt install ghostscript

����ռ����ݸ�ʽת������ GDAL�����룬δ��װ���޷�ʹ�ø߾��ȵ������ݣ�:
$ sudo apt install gdal-bin

���� GIF ��ʽ�Ķ�����Ҫ GraphicsMagick����ѡ��:
$ sudo apt install graphicsmagick

���� MP4��WebM ��ʽ�Ķ�����Ҫ FFmpeg����ѡ��:
$ sudo apt install ffmpeg


# ͨ�� conda ��װ GMT

conda ���� Anaconda �ṩ��һ����ƽ̨�������������conda �� conda-forge Ƶ���ṩ�� GMT ��װ����ʹ�� conda ��װ GMT ���ŵ��У���ƽ̨����װ�򵥡��汾�л�����ȡ�
Anaconda �û�����ֱ��ͨ���������װ�������Լ�ж�� GMT��δ��װ Anaconda ���û����Բο���Anaconda ���׽̡̳���װ Anaconda��
Anaconda �� base ������Ĭ�ϰ�װ�����ٸ�����������в���������� GMT ���ڳ�ͻ������ base ������ GMT ��������б�������⡣Anaconda �û������½�������һ���»�����װʹ�� GMT��
�����Ƽ��û�ʹ��ֻ�ṩ�˱����������� miniconda���Խ�ʡӲ�̿ռ䲢�ұ��� base �����µ� GMT �������⡣


## ��װ GMT

��װ���µ� GMT �ȶ��汾:

$ conda install gmt -c conda-forge

Ҳ���԰�װ GMT �����汾���ÿ����汾��ÿ�����ܸ���һ�Σ�:
$ conda install gmt -c conda-forge/label/dev

## ���԰�װ

$ gmt --version

6.4.0

## ���� GMT

GMT �°汾������ִ�������������� GMT:

$ conda update gmt

## ж�� GMT

ִ�������������ж�� GMT:

$ conda remove gmt
