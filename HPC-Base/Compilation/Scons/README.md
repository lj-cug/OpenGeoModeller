# Scons���

��ȫ����Python�ű����Եı���ϵͳ.

scons��Sconstruct ��Ϊ��ڣ�������ν��б��������
Sconstruct ������һ��python�ļ�������Ҫ��ѭpython���﷨���Լ���ʹ��һЩpython�ķ���.

�ҽӴ��ĳ����У�ֻ��sam(oa)2Դ����Ҫʹ��SCONs����.

## ����

### �򵥵�CPP����

hello.cpp
#include <iostream>
int main() {
    std::cout << "hello world" << std::endl;
}

### Sconstruct

Program("hello.cpp")

Program��Scons�е�һ�����뷽��(builder_method)�� ����Scons ������Ҫ��hello.cpp �����һ����ִ���ļ�.
��֤Sconstruct ��hello.cpp 

��ͬһ���ļ����£�ִ��scons���Ϳ�����ɱ��룬���ɿ�ִ���ļ�hello��
���Կ���������ָֻ����һ��cpp�ļ���scons��Ĭ�ϸ���ִ���ļ�һ�����֣��Լ����.o�ļ������ɣ��ǳ����ܡ���Ȼ������Ҳ��ָ������ļ������֣�Program("target_name", hello.cpp")

���⣬����Program �����������ܶ�builder_method�� ��Object, SharedLibrary��StaticLibrary��LoadableModule��StaticObject��CFile

### sconscript

### ִ�й����ű�

ִ����� scons
