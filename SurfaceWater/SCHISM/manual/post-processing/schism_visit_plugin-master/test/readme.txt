

-- build in Windows and MS Visual Studio
this test code were build in MS Visual studio to debug SCHISM NetCDF output class used in the plugin.

Those plugin header files need to be added to the project,

 "SCHISMFile10.h"
 "MesHconstants10.h"
 "SCHISMFileUtil10.h"
 "SchismGeometry10.h"
 "SCHISMMeshProvider10.h"
 "ZCoordFileMeshProvider10.h"
 "NetcdfSCHISMOutput10.h"
 
And netcdf_c++.lib and its path needed in the project linker option.

-- build in HPC4 linux

   - load cmake and intel compiler 
   - run cmake to generate build system from /test folder
 
     here is a example 

    CC=icc CXX=icpc cmake -S . -B . -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPLUGIN_DIR=/scratch/dms/qshu/visit_3.3.1_plugin_build/unstructure_data/ -DVISIT_DIR=/opt/visit_3.3.1/3.3.1/linux-x86_64/

    where CMAKE Var PLUGIN_DIR is the source folder of plugin unstructure data folder, VISIT_DIR is VISIT installation folder, make sure VISIT_DIR/lib contains netcdf and netcdf_c++ libs,
    
   - then run make

    If you get error of lib linking error like  below,

      ld: cannot find -lnetcdf_c++
      ld: cannot find -lnetcdf
      ld: cannot find -lhdf5_hl
      ld: cannot find -lhdf5
      ld: cannot find -lsz

   Currently you can circumvent this error by modifying link.txt in ./test/test_output/CMakeFiles/test_output.dir, insert lib directory linking option before netcdf link parts like below:

    "  -Wl,-Bstatic -L/opt/visit/3.3.1/3.3.1/linux-x86_64/lib -lnetcdf_c++ -lnetcdf "






