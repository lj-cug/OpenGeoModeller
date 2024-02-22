cmake_minimum_required(VERSION 3.3)
project(Mandelbulb CXX C)

enable_language(CXX)
find_package(ParaView 5.9 REQUIRED COMPONENTS Catalyst PythonCatalyst)
if(NOT PARAVIEW_USE_MPI)
  message(SEND_ERROR "ParaView must be built with MPI enabled")
endif()
find_package(MPI REQUIRED)
include_directories(${MPI_C_INCLUDE_PATH})

add_subdirectory(bin)
