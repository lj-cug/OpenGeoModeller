# training-examples
This is a small set of training exercises used during ESMF and NUOPC tutorials.

* ESMFField -- work with ESMF_Grid and ESMF_Field objects
* ShallowWater -- create a basic NUOPC cap for a shallow water model

## System Setup

Use the following modules to load ESMF on the given machine.

### Theia
```
module use /home/emc.nemspara/SOFT/modulefiles
module load intel/15.1.133 impi/5.1.1.109 netcdf/4.3.0 yaml-cpp esmf/8.0.0bs38
```

### Cheyenne
```
module load intel/19.0.2
export ESMFMKFILE=/glade/work/dunlap/ESMF-INSTALL/8.0.0bs38/lib/libg/Linux.intel.64.mpt.default/esmf.mk
```
