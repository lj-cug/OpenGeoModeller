# Build SCOTH

## Creating the "Makefile.inc" file

cp Make.inc/Makefile.inc.x86-64_pc_linux2 Makefile.inc

## Integer size issues

To coerce the size of the Scotch integer type to 32 or 64 bits, add the "-DINTSIZE32"
or "-DINTSIZE64" flags, respectively, to the C compiler flags in the Makefile.inc configuration file.

在src/Makefile.inc中关闭 **-DPTSCOTCH_PTHREAD** 参数

make libscotch

make libptscotch

