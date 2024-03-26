# Simple_automake

�ο�[����](https://zhuanlan.zhihu.com/p/518876706)

## ��װ
```
# OSX
$ brew install autoconf automake libtool
# Ubuntu/Debian
$ sudo apt-get install autoconf automake libtool
# RHEL/CentOS
$ sudo yum install autoconf automake libtool
```

## ����3���ļ�

configure.ac�� http://Makefile.am �ͳ����� hello.c

## ����
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

configure.ac ���﷨�� MACRO_NAME([param-1],[param-2]..).���ݸ���Ĳ��������÷���������(��������һ��Ҫ�ڵ����ⲿ��֮ǰչ���ĺ꣬��������ǳ�����)���꽫չ��Ϊִ��ʵ�ʼ��� shell �ű����������� configure.ac �ļ��б�д shell �ű���ֻ��һ��������Ӧ��ʹ�� if test < expression >;then... ������ if [[ < expression > ]] ; then... ������������֧����Ϊ�����Żᱻ autoconf ��ϵͳչ����

AC_INIT(package, version, [bug-report], [tarname], [url]);��ÿ��autoconf���ýű��У������������������ʼ��autoconf�����ܺ��԰���ÿ�������ķ����š�

AC_CONFIG_SRCDIR(dir):����������ָ��һ��Ψһ���ļ�����ʶ��������ȷ��Ŀ¼�С�����һ����ȫ��飬�Է��û���д -srcdir ������ѡ�

AC_CONFIG_AUX_DIR(dir) ��ȱʡ����£�autoconf ��������ศ���ļ������������ͷַ����򡣵��ǣ����ǲ�ϣ����Щ�ļ���������Ŀ��Ŀ¼���ڹ����У����ǳ������Ϊ[ build-aux ] �����������Щ������ļ�����build-aux/������ project home �С�

AM_INIT_AUTOMAKE([options]) ��ʼ���Զ�����������һ����Ҫ��ע�������������Ŀ���������ڽ׶Σ���������Ҫ�ṩ��ʼ��automake��ѡ��:foreign�����û���ṩforeign, automake�ᱧԹ�����Ŀû�з���gnu�����׼���⽫Ҫ��������Ŀ����Ŀ¼����README��ChangLog��AUTHORS����������ļ���

AC_PROG_CC ���һ����Ч�� c ���������������������ַ������ļ�顣

AC_CONFIG_FILES(files) �Զ���������ļ�������ļ����������Ǽ򵥵ؽ�Makefile���롣�й���ϸ��Ϣ����鿴�ĵ�automake��

AC_OUTPUT �������ýű�

## Makefile.am

Automake �ļ� http://Makefile.am �� Makefile ����չ��
�����Ա�д��׼�� make �﷨����ͨ��ֻ�趨�����ͳһ����ģʽ�ı����ο���
����ƪ�����У���ֻ�����һ�����ԵĽ��ͣ�������һƪ��������ϸ���ܡ�

bin_PROGRAMS = hello �����һ����Ϊ hello �� PROGRAM (����ѡ����� LIBRARY�� HEADER�� MAN ��) ��������װ�� bin Ŀ¼��(Ĭ��Ϊ/usr/local/bin�����ڵ���/configureʱ������������

hello_SOURCES = hello.c �����Դ������ hello.c

�����ĳ���������ҵ� github �洢�����ҵ�: Example 1.

## More make targets

�� Autoconf �� automake ���ɵ� Makefile �и���������������:

```
make all ����: ����,��,�ĵ���(��make һ��)
make install ��װ��Ҫ��װ���ļ������ļ��Ӱ��������Ƶ�ϵͳ��Χ��Ŀ¼��
make install-strip ��make installһ������Ϊ��Ȼ��ȥ�����Է��š�һЩ�û�ϲ���ÿռ����������õĴ��󱨸档
make uninstall �� make install �෴: �����Ѱ�װ���ļ���(����Ҫ���Ѱ�װ�Ĺ��������С�Ҳ����˵����ͬ��Ŀ¼)
make clean �ӹ���������� make all �������ļ���
make maintainer-clean ������ autoconf ���ɵ��ļ���
make distclean �����ɾ������./configure�������ļ���
make check ���в����׼�������еĻ���
make installcheck ��鰲װ�ĳ��������ȷ��,���֧�ִ����ԵĻ���
make dist ������Դ�ļ������´��� package-version.tar.gz��
���ҵ�һ�ε�����Ӧ��Ϊ���Լ�����Ŀѡ��ʲô���Ĺ���ϵͳʱ���Ҿ��������������Ʒ,���� autoconf ��ʱ������ʹ�á�����һ��������ȷ�ģ�������Խ�����о�����Խ���� autoconf �Ƕ�ôǿ���������������ģ����ʾ���Ѿ���һ�����Ĺ����ű��ͷǳ�ǿ��������������ೣ���������make dist�����İ�ֻ��Ҫһ����С�� unix ���ݻ���(shell �� make)�Ϳ������С�
```