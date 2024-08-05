cmake_minimum_required(VERSION 3.13)
project(Mandelbrot LANGUAGES C CXX)
enable_language(CUDA)

include (GNUInstallDirs)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")

#------------------------------------------------------------------------------
# since we use C++11 in this example.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#------------------------------------------------------------------------------
add_executable(Mandelbrot
  Mandelbrot.cu
  Mandelbrot.h
  driver.cxx)
target_compile_options(Mandelbrot PRIVATE "-lineinfo")
set_target_properties(Mandelbrot PROPERTIES CUDA_ARCHITECTURES OFF)

#------------------------------------------------------------------------------
option(USE_CATALYST "Build example with Catalyst enabled" ON)
if (USE_CATALYST)
  find_package(catalyst REQUIRED
    PATHS "${ParaView_DIR}/catalyst")
  target_compile_definitions(Mandelbrot
    PRIVATE
      USE_CATALYST=1)
  target_link_libraries(Mandelbrot
    PRIVATE
      catalyst::catalyst)
endif()
