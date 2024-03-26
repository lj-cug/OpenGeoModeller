# RegCM

The **ATM_SRC** environment variable is used to point the installation directory of the atmospheric model and user might replace it by any valid directory name.
The command that is given here is for Intel Compiler and Linux operating system and might change in different working environment amd compiler. By design, the model components do not have any ESMF related code and user does not need to use ESMF library in the installation of the individual model components. All ESMF related code is placed in "driver" side (RegESM). To that end, the installation of the model is almost same as the standalone version except given extra configure option (**--enable-cpl**) to enable coupling support.

The additional configuration parameters (see Doc/README.namelist in the RegCM source directory) for the RegCM need to be activated when the model is running in the coupled mode and the interaction with other model components such as ocean, wave or both of them. In this case, the user need to use and turn on coupling related options in the RegCM configuration file.

**Physics parameters:**

```
&physicsparam
...
iocncpl = 1,
iwavcpl = 1,
...
```

In this case, **iocncpl** activates coupling with ocean model (ROMS or MITgcm) and **iwavcpl** coupling with wave model (WAM). In case of coupling both ocean and wave model, both options must be activated.

**Coupling parameters:**

```
&cplparam
cpldt = 10800.,
zomax = 0.005,
ustarmax = 1.2,
/ 
```

The coupling interval (in seconds) given with **cpldt** option must be consistent with the coupling interval defined in the driver namelist file (**namelist.rc**). Parameter **zomax** is the threshold for surface roughness. In addition to **zomax**, parameter **ustarmax** is the threshold for frictional velocity. In this case, both parameters are only valid for wave coupling. The threshold values for surface roughness (**zomax**) and frictional velocity (**ustarmax**) are used to keep atmospheric model stable and user might need to play with those number to find the reasonable values as large as possible.
