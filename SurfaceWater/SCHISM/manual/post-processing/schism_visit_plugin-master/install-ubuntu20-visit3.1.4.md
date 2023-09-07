# Installing on Ubuntu 20

These instructions were tested on a clean VM with Ubuntu 20 OS. (I used Jetstream2 for cloud services.)

## First, install VisIt

See the VisIt User Guide, [Installing and Starting VisIt](https://visit-sphinx-github-user-manual.readthedocs.io/en/develop/getting_started/Installing_VisIt.html)

We will use VisIt version [3.1.4](https://visit-dav.github.io/visit-website/releases-as-tables/#series-31) so we can use server mode on Expanse.  Get the source code and the install script.
```
wget https://github.com/visit-dav/visit/releases/download/v3.1.4/visit3_1_4.linux-x86_64-ubuntu20.tar.gz
wget https://github.com/visit-dav/visit/releases/download/v3.1.4/visit-install3_1_4
```
Change to executable mode
```
chmod +x visit-install3_1_4
```
For the following, we install visit in the home directory to avoid using sudo.  Choose '1', no specific configuration.
```
./visit-install3_1_4 3.1.4 linux-x86_64-ubuntu20 ~/visit
1
```
Set the path for the executable
```
export PATH=~/visit/bin:$PATH
```

Try VisIt:
```
visit
```
It works.

# Install the plugin

Now try the steps to install the plugin.

Get the plugin code.  This one test *my* plugin code with the malloc.h in ifdef statements:
```
git clone https://github.com/lisalenorelowe/schism_visit_plugin.git
cd schism_visit_plugin
```

Stop it from using Ubuntu compiler:
```
alias c++='g++'
alias cc='gcc'
```

Install the unstructure_data plugin:
```
cd unstructure_data
```
Save a copy of the main VisIt plugin class files to your home directory. These files will be overwritten in the next step by VisIt's code skeleton generating tool.
```
cp avtSCHISMFileFormat.C ~
cp avtSCHISMFileFormat.h ~
```
The command `xml2plugin` is a VisIt command.
```
xml2plugin -clobber SCHISMOutput.xml
```
SCHISMOutput.xml is the file used by VisIt code generating tool to create code skeleton and makelist file.

Use two backup plugin class file to overwrite the files avtSCHISMFileFormat.C and avtSCHISMFileFormat.h.
```
cp ~/avtSCHISMFileFormat.C .
cp ~/avtSCHISMFileFormat.h .
```

Make a build directory:
```
mkdir build
cd build
```

Use `cmake` to create the `make` system.
```
cmake -DCMAKE_BUILD_TYPE:STRING=Release ..
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
cp avtpropFileFormat.C ~
cp avtpropFileFormat.h ~
xml2plugin -clobber prop.xml
cp ~/avtpropFileFormat.C .
cp ~/avtpropFileFormat.h .
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE:STRING=Release ..
make
ls ~/.visit/3.1.4/linux-x86_64/plugins/databases
```

And
```
cd ~/schism_visit_plugin/gr3
cp avtgr3FileFormat.C ~
cp avtgr3FileFormat.h ~
xml2plugin -clobber gr3.xml
cp ~/avtgr3FileFormat.C .
cp ~/avtgr3FileFormat.h .
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
cp avtMDSCHISMFileFormat.C ~
cp avtMDSCHISMFileFormat.h ~
xml2plugin -clobber mdschism.xml
cp ~/avtMDSCHISMFileFormat.C .
cp ~/avtMDSCHISMFileFormat.h .
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

Now try it out.  Links to data are on the [SCHISM visualization page](https://schism-dev.github.io/schism/master/getting-started/visualization.html)

```
wget -r -nH --cut-dirs=2 -np -R "index.html*" http://ccrm.vims.edu/yinglong/SVN_large_files/Scribe_IO_outputs/
wget -r -nH --cut-dirs=2 -np -R "index.html*" http://ccrm.vims.edu/yinglong/SVN_large_files/SCHISM_v5.6.1_sample_outputs/
```

For SCHISM_v5.6.1_sample_outputs:
![](READ_OLDIO.png)

It works:
![](OLD_IO.png)


When I use the new IO, I need to specify "SCHISM" when reading in the file, and it has an artifact that makes it impossible to do the transform.
![](artifact.png)


