# Meson��������

Mesonּ�ڿ�����߿����ԺͿ��ٵĹ���ϵͳ���ṩ�򵥵�ǿ�������ʽ������������������ԭ��֧�����µĹ��ߺͿ�ܣ���Qt5�����븲���ʡ���Ԫ���Ժ�Ԥ����ͷ�ļ��ȡ�����һ���Ż����������ٱ�����룬���������������ȫ���롣

## ubuntu�ϰ�װmeson
   
    ��װpython3��ninja:

	apt-get install python3 python3-pip ninja-build

	pip install meson ninja
	
	��meson.buildĿ¼ִ�У� 	
	meson build
	
	���� buildĿ¼��ִ��ninja
	
	cd ninja && ninja
	
## meson-ui (MESON���ӻ�����)

Qt GUI for the Meson build system

https://github.com/michaelbrockus/meson-ui

pip install meson-ui

## meson.build��д

�����ִ�г���

project('project01', 'c')
executable("project", 'src/main.c')

���뾲̬���ӿ⣺

project('project02', 'c')
static_library('thirdinfo', 'src/third_lib.c')


