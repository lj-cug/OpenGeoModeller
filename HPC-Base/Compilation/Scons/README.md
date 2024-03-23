# Scons简介

完全基于Python脚本语言的编译系统.

scons由Sconstruct 作为入口，控制如何进行编译操作。
Sconstruct 本身是一个python文件，故需要遵循python的语法，以及能使用一些python的方法.

我接触的程序中，只有sam(oa)2源码需要使用SCONs编译.

## 举例

### 简单的CPP程序

hello.cpp
#include <iostream>
int main() {
    std::cout << "hello world" << std::endl;
}

### Sconstruct

Program("hello.cpp")

Program是Scons中的一个编译方法(builder_method)， 告诉Scons 我们想要把hello.cpp 编译成一个可执行文件.
保证Sconstruct 和hello.cpp 

在同一个文件夹下，执行scons，就可以完成编译，生成可执行文件hello。
可以看到，我们只指定了一个cpp文件，scons会默认给可执行文件一个名字，以及完成.o文件的生成，非常智能。当然，我们也能指定输出文件的名字，Program("target_name", hello.cpp")

另外，除了Program ，还有其他很多builder_method， 如Object, SharedLibrary，StaticLibrary，LoadableModule，StaticObject，CFile

### sconscript

### 执行构建脚本

执行命令： scons
