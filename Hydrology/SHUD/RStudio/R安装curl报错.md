# ���R��װcurl�Ĵ���

�ο��� https://blog.csdn.net/qq_34536369/article/details/131579825

## ��װ openssl

apt install libcurl4-openssl-dev

## �ֶ�ָ��INCLUDE_DIR��LIB_DIR

�ȵ��ҵ�libcurl.pc��λ��

����ֱ��sudo find / -name "libcurl.pc"������е���

ֱ����Ĭ��Ӧ������/usr/lib/x86_64-linux-gnu/pkgconfig/libcurl.pc

## ����Renviron

����gedit ~/.Renviron

���Ǹ����ļ���д�룺

INCLUDE_DIR=/usr/lib/x86_64-linux-gnu/pkgconfig/libcurl.pc
LIB_DIR=/usr/lib/x86_64-linux-gnu/pkgconfig/libcurl.pc

Ȼ���ٴ�Rȥװcurl�ͳɹ���
