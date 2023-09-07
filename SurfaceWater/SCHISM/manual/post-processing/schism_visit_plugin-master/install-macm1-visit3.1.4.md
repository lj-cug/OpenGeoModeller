# Installing on Mac Studio

Instructions for installing the SCHISM plugins for VisIt 3.1.4 on a Mac Studio that has an M1 chip.

We are using VisIt 3.1.4 to use client-server mode from Expanse.  The local VisIt(client) must use the same VisIt as the HPC(server). Older versions of VisIt(pre-M1) are only available as x86, so an extra CMAKE flag is needed to compile for x86 instead of the default arm(M1) processor.


## Install VisIt 3.1.4

First, install VisIt 3.1.4 in the usual Mac way: [download](https://visit-dav.github.io/visit-website/releases-as-tables/#series-31), click, drag to applications. Test it by clicking on the application icon to open VisIt, then close VisIt.

To follow the instructions below, VisIt must be in the path.  If you open a terminal and type `visit`, it will probably say `command not found: visit`.  The VisIt executable is usually here: `/Applications/VisIt.app/Contents/Resources/bin`.  Confirm the following opens VisIt.
```
/Applications/VisIt.app/Contents/Resources/bin/visit
```

To avoid typing the whole filepath, add VisIt to your PATH by doing:
```
export PATH="/Applications/VisIt.app/Contents/Resources/bin:$PATH"
visit
```

That PATH will only be valid while that terminal is open.  To add an additional executable path to your environment, add it to the 'dot' file for your shell, which (unless you changed your default shell), is the file `~/.zshrc`.  Open `~/.zshrc` and add the following, 
```
export PATH="/Applications/VisIt.app/Contents/Resources/bin:$PATH"
```

Exit the terminal and open a new one.  Now, `visit` should be in your path.  The command `xml2plugin` used below is a VisIt command.  To confirm that both are in your path, do:
```
which visit
which xml2plugin
```

If that was successful, you're ready to move on.

## Get CMake

[Download CMake](https://cmake.org/download/) for Mac, drag it into the applications bin.  Open CMake, in the top menu bar, click Tools, 
choose How to Install For Command Line Use.  I chose this way:
```
sudo "/Applications/CMake.app/Contents/bin/cmake-gui" --install
```

## Install the plugins

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

Use `cmake` to create the `make` system.  My mac is arm but the executable for VisIt 3.1.4 is x86_64, so the next command tells it to target x86.
```
cmake -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_OSX_ARCHITECTURES=x86_64 ..
```

Run `make` to build plugins binary. 
```
make
```

Check
```
ls ~/.visit/3.1.4/darwin-x86_64/plugins/databases
```

There should be four new files in ~/.visit/3.1.4/darwin-x86_64/plugins/databases:
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
cmake -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_OSX_ARCHITECTURES=x86_64 ..
make
ls ~/.visit/3.1.4/darwin-x86_64/plugins/databases
```
Works.

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
cmake -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_OSX_ARCHITECTURES=x86_64 ..
make
ls ~/.visit/3.1.4/darwin-x86_64/plugins/databases
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
cmake -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_OSX_ARCHITECTURES=x86_64 ..
make
```
Error.  

So, this is where we hack the makefiles, as per Eric's email, 
```
vi CMakeFiles/EMDSCHISMDatabase_par.dir/flags.make
vi CMakeFiles/MMDSCHISMDatabase.dir/flags.make
vi CMakeFiles/EMDSCHISMDatabase_ser.dir/flags.make
vi CMakeFiles/IMDSCHISMDatabase.dir/flags.make
```
remove all the
```
libnetcdf_c++.dylib libnetcdf.dylib libhdf5_hl.dylib libhdf5.dylib libsz.dylib libz.dylib
```

Then make
```
make
```

Check
```
ls ~/.visit/3.1.4/darwin-x86_64/plugins/databases
```


And here you go:
```
(base) lllowe@LisasMacStudio build % ls ~/.visit/3.1.4/darwin-x86_64/plugins/databases
libEMDSCHISMDatabase_par.dylib libEgr3Database_par.dylib      libIMDSCHISMDatabase.dylib     libMMDSCHISMDatabase.dylib
libEMDSCHISMDatabase_ser.dylib libEgr3Database_ser.dylib      libISCHISMDatabase.dylib       libMSCHISMDatabase.dylib
libESCHISMDatabase_par.dylib   libEpropDatabase_par.dylib     libIgr3Database.dylib          libMgr3Database.dylib
libESCHISMDatabase_ser.dylib   libEpropDatabase_ser.dylib     libIpropDatabase.dylib         libMpropDatabase.dylib
```


Now try it out.  Links to data are on the [SCHISM visualization page](https://schism-dev.github.io/schism/master/getting-started/visualization.html)

```
wget -r -nH --cut-dirs=2 -np -R "index.html*" http://ccrm.vims.edu/yinglong/SVN_large_files/Scribe_IO_outputs/
wget -r -nH --cut-dirs=2 -np -R "index.html*" http://ccrm.vims.edu/yinglong/SVN_large_files/SCHISM_v5.6.1_sample_outputs/
```
