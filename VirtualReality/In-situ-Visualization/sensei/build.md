# build sensei
`
git clone https://github.com/SENSEI-insitu/SENSEI
mkdir sensei-build
cd sensei-build
cmake ../SENSEI
make -j
`

# Cmake options
`
Build Option 				Default            Description
SENSEI_ENABLE_CUDA			OFF 		Enables CUDA accelerated codes. Requires compute capability 7.5 and CUDA 11 or later.
SENSEI_ENABLE_PYTHON		ON 			Enables Python bindings. Requires Python, Numpy, mpi4py, and SWIG.
SENSEI_ENABLE_CATALYST	    OFF 		Enables the Catalyst analysis adaptor. Depends on ParaView Catalyst. Set ParaView_DIR.
SENSEI_ ENABLE_CATALYST_PYTHON	OFF 	Enables Python features of the Catalyst analysis adaptor. ParaView_DIR Set to the directory containing ParaViewConfig.cmake.
SENSEI_ENABLE_ASCENT		OFF 		Enables the Ascent analysis adaptor. Requires an Ascent install.
ASCENT_DIR Set to the directory containing the Ascent CMake configuration.
SENSEI_ENABLE_ADIOS2		OFF			 Enables ADIOS 2 in transit transport. Set ADIOS2_DIR.
ADIOS2_DIR Set to the directory containing ADIOSConfig.cmake
SENSEI_ENABLE_HDF5			OFF			 Enables HDF5 adaptors and endpoints. Set HDF5_DIR.
HDF5_DIR Set to the directory containing HDF5Config.cmake
SENSEI_ENABLE_LIBSIM		OFF 	Enables Libsim data and analysis adaptors. Requires Libsim. Set VTK_DIR and LIBSIM_DIR. LIBSIM_DIR Path to libsim install.
SENSEI_ENABLE_VTK_IO		OFF 		Enables adaptors to write to VTK XML format.
SENSEI_ENABLE_VTK_MPI		OFF 		Enables MPI parallel VTK filters, such as parallel I/O. VTK_DIR Set to the directory containing VTKConfig.cmake.
SENSEI_ENABLE_VTKM			OFF Enables analyses that use VTK-m. Requires an install of VTK-m. Experimental,
each implementation requires an exact version match
`