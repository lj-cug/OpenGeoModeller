# Simple_automake

参考[文献](https://zhuanlan.zhihu.com/p/518876706)

## 安装
```
# OSX
$ brew install autoconf automake libtool
# Ubuntu/Debian
$ sudo apt-get install autoconf automake libtool
# RHEL/CentOS
$ sudo yum install autoconf automake libtool
```

## 创建3个文件

configure.ac、 http://Makefile.am 和程序本身 hello.c

## 编译
```
# this creates the configure script
$ autoreconf --verbose --install --force
$ ./configure --help
$ ./configure
ecking for a BSD-compatible install... /usr/bin/install -c
checking whether build environment is sane... yes
checking for a thread-safe mkdir -p... build-aux/install-sh -c -d
checking for mawk... no
...
config.status: creating Makefile
config.status: executing depfiles commands
# Now try the makefile
$ make
gcc -DPACKAGE_NAME=\"hello\" -DPACKAGE_TARNAME=\"hello\" -DPACKAGE_VERSION=\"1.0\" -DPACKAGE_STRING=\"hello\ 1.0\" -DPACKAGE_BUGREPORT=\"\" -DPACKAGE_URL=\"\" -DPACKAGE=\"hello\" -DVERSION=\"1.0\" -I.     -g -O2 -MT hello.o -MD -MP -MF .deps/hello.Tpo -c -o hello.o hello.c
mv -f .deps/hello.Tpo .deps/hello.Po
gcc  -g -O2   -o hello hello.o
# We now have the hello program built
$ ./hello
hello world!
```

## configure.ac

configure.ac 的语法是 MACRO_NAME([param-1],[param-2]..).传递给宏的参数必须用方括号引用(除非是另一个要在调用外部宏之前展开的宏，这种情况非常罕见)。宏将展开为执行实际检查的 shell 脚本。还可以在 configure.ac 文件中编写 shell 脚本。只有一个区别，您应该使用 if test < expression >;then... 而不是 if [[ < expression > ]] ; then... 来进行条件分支，因为方括号会被 autoconf 宏系统展开。

AC_INIT(package, version, [bug-report], [tarname], [url]);在每个autoconf配置脚本中，你必须首先用这个宏初始化autoconf。不能忽略包含每个参数的方括号。

AC_CONFIG_SRCDIR(dir):接下来我们指定一个唯一的文件，标识我们在正确的目录中。这是一个安全检查，以防用户重写 -srcdir 命令行选项。

AC_CONFIG_AUX_DIR(dir) 在缺省情况下，autoconf 将创建许多辅助文件来帮助构建和分发程序。但是，我们不希望这些文件混乱了项目主目录。在惯例中，我们称这个宏为[ build-aux ] ，因此它将这些额外的文件放在build-aux/而不是 project home 中。

AM_INIT_AUTOMAKE([options]) 初始化自动化。这里有一个重要的注意事项，在您的项目开发的早期阶段，您可能想要提供初始化automake的选项:foreign。如果没有提供foreign, automake会抱怨你的项目没有符合gnu编码标准，这将要求你在项目的主目录中有README、ChangLog、AUTHORS和许多其他文件。

AC_PROG_CC 检查一个有效的 c 编译器。你可以在这个部分放入更多的检查。

AC_CONFIG_FILES(files) 自动创建输出文件所需的文件。这里我们简单地将Makefile放入。有关详细信息，请查看文档automake。

AC_OUTPUT 创建配置脚本

## Makefile.am

Automake 文件 http://Makefile.am 是 Makefile 的扩展。
您可以编写标准的 make 语法，但通常只需定义符合统一命名模式的变量参考。
在这篇文章中，我只会给出一个粗略的解释，并在下一篇文章中详细介绍。

bin_PROGRAMS = hello 输出是一个名为 hello 的 PROGRAM (其他选项包括 LIBRARY、 HEADER、 MAN 等) ，将被安装在 bin 目录中(默认为/usr/local/bin，但在调用/configure时可以配置它。

hello_SOURCES = hello.c 程序的源代码是 hello.c

完整的程序可以在我的 github 存储库中找到: Example 1.

## More make targets

由 Autoconf 和 automake 生成的 Makefile 有更多的命令可以运行:

```
make all 构建: 程序,库,文档等(与make 一样)
make install 安装需要安装的文件，将文件从包的树复制到系统范围的目录。
make install-strip 与make install一样的行为，然后去掉调试符号。一些用户喜欢用空间来交换有用的错误报告。
make uninstall 与 make install 相反: 擦除已安装的文件。(这需要从已安装的构建树运行。也就是说在相同的目录)
make clean 从构建树中清除 make all 构建的文件。
make maintainer-clean 擦除由 autoconf 生成的文件。
make distclean 额外地删除所有./configure创建的文件。
make check 运行测试套件，如果有的话。
make installcheck 检查安装的程序或库的正确性,如果支持此特性的话。
make dist 从所有源文件中重新创建 package-version.tar.gz。
当我第一次调查我应该为我自己的项目选择什么样的构建系统时，我经常看到其他替代品,声称 autoconf 过时且难以使用。这有一部分是正确的，但是我越深入研究，就越发现 autoconf 是多么强大。正如您所看到的，这个示例已经用一个简洁的构建脚本和非常强大的输出覆盖了许多常见的情况。make dist创建的包只需要一个最小的 unix 兼容环境(shell 和 make)就可以运行。
```