# ESMX Application Prototypes
The ESMX application prototypes demonstrate model coupling using the Earth System Modeling Executable (ESMX) package.

The [ESMX](https://github.com/esmf-org/esmf/tree/develop/src/addon/ESMX) layer is built on top of ESMF and NUOPC. The idea is to make it as simple as possible for a user to build, run, and test NUOPC based systems, often without having to write any extra code beyond the NUOPC-compliant models.

To build the Basic Application (apps/basicApp.yaml) with debugging (-g) and run all tests (-t) execute the ESMX Builder with the following command line options.
```
  ESMX_Builder apps/basicApp.yaml -g -t
```
