# Installing on Expanse (SDSC)

VisIt 3.1.4 is installed on Expanse.  You can install the plugin in your home directory.

When using client-server mode, the local VisIt(client) must use the same VisIt as the HPC(server).  To use Expanse client-server VisIt (as of 6/30/2023), you must also install this plugin on your local machine with VisIt 3.1.4. 

## Set the VisIt environment

Currently, there is no 'visit' module, so we need to set the environment by adding the path to VisIt's xml commands used to compile the plugins.

The instructions for running in client-server mode are here: `/cm/shared/examples/sdsc/visit/README`.

From that, you can get the path: `/cm/shared/apps/vis/visit/3.1.4/gcc/9.2.0/openmpi/3.1.6/3.1.4/linux-x86_64/bin`, which indicates that VisIt 3.1.4 was compiled with gcc 9.2.0 and openmpi 3.1.6.

First, load those gcc and openmpi modules, and load cmake.
```
module load gcc/9.2.0
module load openmpi/3.1.6
module load cmake
```

Here is my modulefile for VisIt 3.1.4.
```
#%Module
set VISITARCHHOME "/cm/shared/apps/vis/visit/3.1.4/gcc/9.2.0/openmpi/3.1.6/3.1.4/linux-x86_64"
setenv VISITARCHHOME "/cm/shared/apps/vis/visit/3.1.4/gcc/9.2.0/openmpi/3.1.6/3.1.4/linux-x86_64"
prepend-path PATH "$VISITARCHHOME/bin"
prepend-path LD_LIBRARY_PATH "$VISITARCHHOME/lib"
setenv VISITPLUGININSTPRI "/home/llowe/.visit/3.1.4/linux-x86_64/plugins/"
```

And I have this in my .bashrc:
```
module use --append /home/llowe/modulefiles
```

So now load visit
```
module load visit
```

If you don't want to make a module, do:
```
export PATH=/cm/shared/apps/vis/visit/3.1.4/gcc/9.2.0/openmpi/3.1.6/3.1.4/linux-x86_64/bin:$PATH
```

## Install the plugin

Now try the steps to install the plugin.

Get the plugin code.
```
git clone https://github.com/schism-dev/schism_visit_plugin.git
cd schism_visit_plugin
```

Install the unstructure_data plugin:
```
cd unstructure_data
```

The commands with `xml2` are VisIt commands.
```
xml2cmake -clobber SCHISMOutput.xml
xml2info SCHISMOutput.xml
```
SCHISMOutput.xml is the file used by VisIt code generating tool to create code skeleton and makelist file.

Make a build directory:
```
mkdir build
cd build
```

Use `cmake` to create the `make` system.
```
cmake -DCMAKE_BUILD_TYPE:STRING=Release -S .. -B .
```

Run `make` to build plugins binary. 
```
make
```
Check:
```
ls ~/.visit/3.1.4/linux-x86_64/plugins/databases
```
There should be four new files in ~/.visit/3.1.4/linux-x86_64/plugins/databases:
```
libESCHISMDatabase_par.so
libESCHISMDatabase_ser.so
libISCHISMDatabase_par.so
libMSCHISMDatabase_par.so
```

Repeat the steps for the other plugins:
```
cd ~/schism_visit_plugin/prop
xml2cmake -clobber prop.xml
xml2info prop.xml
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE:STRING=Release -S .. -B .
make
ls ~/.visit/3.1.4/linux-x86_64/plugins/databases
```

And
```
cd ~/schism_visit_plugin/gr3
xml2cmake -clobber gr3.xml
xml2info gr3.xml
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE:STRING=Release ..
make
ls ~/.visit/3.1.4/linux-x86_64/plugins/databases
```
Works.

And
```
cd ~/schism_visit_plugin/mdschism
xml2cmake -clobber mdschism.xml
xml2info mdschism.xml
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE:STRING=Release ..
make
```
Error.

So, this is where we hack the makefiles, as per Eric's email (see below), 
```
vi CMakeFiles/EMDSCHISMDatabase_par.dir/flags.make
vi CMakeFiles/MMDSCHISMDatabase.dir/flags.make
vi CMakeFiles/EMDSCHISMDatabase_ser.dir/flags.make
vi CMakeFiles/IMDSCHISMDatabase.dir/flags.make
```
and remove all the
```
libnetcdf_c++.a libnetcdf.a libhdf5_hl.so libhdf5.so libsz.so libz.so
```
and then, do make:
```
make
```
Check that you have all the plugins now:
```
ls ~/.visit/3.1.4/linux-x86_64/plugins/databases
```
You should have the following.
```
libEMDSCHISMDatabase_par.so  libEpropDatabase_par.so  libMMDSCHISMDatabase.so
libEMDSCHISMDatabase_ser.so  libEpropDatabase_ser.so  libMSCHISMDatabase.so
libESCHISMDatabase_par.so    libIMDSCHISMDatabase.so  libMgr3Database.so
libESCHISMDatabase_ser.so    libISCHISMDatabase.so    libMpropDatabase.so
libEgr3Database_par.so	     libIgr3Database.so
libEgr3Database_ser.so	     libIpropDatabase.so
```
