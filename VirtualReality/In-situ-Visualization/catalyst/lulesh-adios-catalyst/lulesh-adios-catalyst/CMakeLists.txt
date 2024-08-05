cmake_minimum_required(VERSION 3.10)
project(lulesh CXX C)
set(CMAKE_CXX_STANDARD 11)

include (GNUInstallDirs)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")

option(WITH_OPENMP    "Build LULESH with OpenMP"       TRUE)
option(WITH_MPI       "Build LULESH with MPI"       TRUE)

set(INSITU None CACHE STRING "Enable in-situ support")
set_property(CACHE INSITU PROPERTY STRINGS None Catalyst Ascent)

if (WITH_OPENMP)
  find_package(OpenMP REQUIRED)
endif()
if (WITH_MPI)
  find_package(MPI COMPONENTS CXX)
endif()

set(LULESH_SOURCES
  lulesh-comm.cc
  lulesh-init.cc
  lulesh-util.cc
  lulesh-viz.cc
  lulesh.cc
  )
  
add_executable(lulesh2.0 ${LULESH_SOURCES})
target_include_directories(lulesh2.0 PUBLIC
                           "${PROJECT_SOURCE_DIR}"
                           )
target_compile_definitions(lulesh2.0
	    PRIVATE
	    "USE_MPI=1"
	    )
target_link_libraries(lulesh2.0 PRIVATE MPI::MPI_CXX)

if(INSITU STREQUAL "Catalyst")
	find_package(catalyst REQUIRED PATHS "$ENV{EBROOTPARAVIEW}/lib/cmake/paraview-5.10")
	target_compile_definitions(lulesh2.0
	    PRIVATE
	    "PARAVIEW_IMPL_DIR=\"${ParaView_DIR}\""
	    VIZ_CATALYST=1
	    )
    target_link_libraries(lulesh2.0 PRIVATE catalyst::catalyst)
elseif(INSITU STREQUAL "Ascent")
    find_package(Ascent REQUIRED PATHS "/apps/daint/UES/Ascent/ascent-install/lib/cmake/ascent")
    target_compile_definitions(lulesh2.0
	    PRIVATE
	    VIZ_ASCENT=1
	    )
    target_link_libraries(lulesh2.0 PRIVATE ascent::ascent_mpi)
endif()
