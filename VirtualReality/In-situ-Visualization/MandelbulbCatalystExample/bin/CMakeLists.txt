set(Adaptor_SRCS
    ${CMAKE_SOURCE_DIR}/src/InSituAdaptor.cpp
    )
add_library(CatalystMandelbulbAdaptor ${Adaptor_SRCS})
target_link_libraries(CatalystMandelbulbAdaptor ParaView::PythonCatalyst VTK::ParallelMPI VTK::CommonDataModel)

add_executable(Mandelbulb 
     ${CMAKE_SOURCE_DIR}/src/Mandelbulb.cpp)

target_link_libraries(Mandelbulb LINK_PRIVATE CatalystMandelbulbAdaptor)
