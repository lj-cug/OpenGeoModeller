# 3.3. Installation of RegESM

After installation of the individual model components such as RegCM, ROMS or MITgcm, the RegESM driver can be installed. The RegESM basically uses the object (\*.o), module (\*.mod) and header (\*.h) files of individual model components to create static library files such as libatm.a for atmosphere, libocn.a for ocean, librtm.a for river routing, libwav.a for wave model and libcop.a for co-processing component. The static libraries are used to create the single RegESM executable that able to run model components.

The ESMF library is required to compile ESMF related component codes in "driver" side. The configuration script basically tries to find the installation directory of ESMF library by looking for a specific environment variable (**ESMF\_LIB**). If **ESMF\_LIB** environment variable points the directory of ESMF shared library (libesmf.so) and configuration file (esmf.mk), then it uses them to compile the RegESM. In case of undefined **ESMF\_LIB** environment variable, user might specify the ESMF library directory by using **--with-esmf** configure option.

Currently, RegESM project is maintained and distributed by using a [GitHub repository](https://github.com/uturuncoglu/RegESM) under GNU GPL license. The documentation can be found under the [docs/](https://github.com/uturuncoglu/RegESM/tree/master/docs) directory. To open new issue ticket, bug report or feature request that are related with the driver, the GitHub page can be used. Due to the limitation of the human resource to develop and maintain the coupled modeling system, the solution of the possible bugs and issues can be delayed.

To install RegESM with four component (ATM, OCN, RTM and WAV):

```
cd $ESM_SRC
git clone https://github.com/uturuncoglu/RegESM.git
cd RegESM
./bootstrap.sh
./configure --prefix=$ESM_SRC/RegESM --with-atm=$ATM_SRC --with-ocn=$OCN_SRC/Build --with-rtm=$RTM_SRC ¨Cwith-wav=$WAV_SRC/obj CC=icc FC=ifort
make
make install
```

If RegESM model is used with fewer components then the options given in the configure script can be modified based on the selected model components such as ATM-OCN, ATM-OCN-RTM, ATM-WAV. The configure options **--with-atm**, **--with-ocn**, **--with-rtm** and **--with-wav** are used to point the installation directories of the model components. For ROMS case, **--with-ocn** option must point "Build" directory that holds the compiled source files of the ROMS installation but "build" directory for MITgcm. For the wave component (WAV) "obj" directory must be given to **--with-wav**.

The configure script is smart enough to check some key files to find the correct ocean model component (ROMS, ROMS with sea-ice support or MITgcm) and compiles the required files suitable for selected model components. In addition, the configure script also checks the ROMS model installation directory to enable the sea-ice related part of the data exchange routines in the "driver" side. To that end, user does not need to set any other option when using sea-ice enabled ROMS version.

Also note that **$ESM\_SRC** is the main directory for RegESM installation.