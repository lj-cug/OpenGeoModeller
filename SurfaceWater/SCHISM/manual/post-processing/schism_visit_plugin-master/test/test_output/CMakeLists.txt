set(PROJECT_NAME test_output)
#set(PLUGIN_DIR "/scratch/dms/qshu/visit_3.3.1_plugin_build/unstructure_data/" )
#set(VISIT_DIR "/scratch/dms/qshu/visit_3.3.1/3.3.1/linux-x86_64")
INCLUDE("${VISIT_DIR}/include/PluginVsInstall.cmake")
INCLUDE("${VISIT_DIR}/include/VisItLibraryDependencies.cmake")

################################################################################
# Source groups
################################################################################
#target_include_directories(${PROJECT_NAME} PRIVATE ${PLUGIN_DIR})
include_directories(${PLUGIN_DIR})
include_directories(${NETCDF_INCLUDE_DIR})

set(Source_Files "test_output.cpp"
            ${PLUGIN_DIR}/SCHISMFile10.h
            ${PLUGIN_DIR}/MeshConstants10.h
            ${PLUGIN_DIR}/MeshProvider10.h
            ${PLUGIN_DIR}/SCHISMFileUtil10.h
            ${PLUGIN_DIR}/SchismGeometry10.h
            ${PLUGIN_DIR}/SCHISMMeshProvider10.h
            ${PLUGIN_DIR}/ZCoordFileMeshProvider10.h
            ${PLUGIN_DIR}/NetcdfSCHISMOutput10.h
            ${PLUGIN_DIR}/MeshProvider10.cpp
            ${PLUGIN_DIR}/SCHISMFile10.cpp
            ${PLUGIN_DIR}/SCHISMFileUtil10.cpp
            ${PLUGIN_DIR}/SchismGeometry10.cpp
            ${PLUGIN_DIR}/SCHISMMeshProvider10.cpp
            ${PLUGIN_DIR}/ZCoordFileMeshProvider10.cpp
            ${PLUGIN_DIR}/NetcdfSCHISMOutput10.cpp)


#set(Source_Files
#    "test_output.cpp"
#)

source_group("Source Files" FILES ${Source_Files})

set(ALL_FILES ${Source_Files})
message("all files ${ALL_FILES}")
################################################################################
# Target
################################################################################
add_executable(${PROJECT_NAME} ${ALL_FILES})

#use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "${DEFAULT_CXX_PROPS}")
set(ROOT_NAMESPACE testnclib)

set_target_properties(${PROJECT_NAME} PROPERTIES
    VS_GLOBAL_KEYWORD "Win32Proj"
)
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE "TRUE"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE "TRUE"
    )
endif()
################################################################################
# Compile definitions
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        "$<$<CONFIG:Debug>:"
            "_DEBUG"
        ">"
        "$<$<CONFIG:Release>:"
            "NDEBUG"
        ">"
        "_CONSOLE;"
        "UNICODE;"
        "_UNICODE"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        "$<$<CONFIG:Debug>:"
            "_DEBUG"
        ">"
        "$<$<CONFIG:Release>:"
            "NDEBUG"
        ">"
        "WIN32;"
        "_CONSOLE;"
        "UNICODE;"
        "_UNICODE"
    )
endif()

################################################################################
# Compile and link options
################################################################################
if(MSVC)
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Debug>:
                /Od
            >
            $<$<CONFIG:Release>:
                /O2;
                /Oi;
                /Gy
            >
            /permissive-;
            /sdl;
            /W3;
            ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
            ${DEFAULT_CXX_EXCEPTION_HANDLING};
            /Y-
        )
    elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
        target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Debug>:
                /Od
            >
            $<$<CONFIG:Release>:
                /O2;
                /Oi;
                /Gy
            >
            /permissive-;
            /sdl;
            /W3;
            ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
            ${DEFAULT_CXX_EXCEPTION_HANDLING};
            /Y-
        )
    endif()
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        target_link_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Debug>:
                /INCREMENTAL
            >
            $<$<CONFIG:Release>:
                /OPT:REF;
                /OPT:ICF;
                /INCREMENTAL:NO
            >
            /DEBUG;
            /SUBSYSTEM:CONSOLE
        )
    elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
        target_link_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Debug>:
                /INCREMENTAL
            >
            $<$<CONFIG:Release>:
                /OPT:REF;
                /OPT:ICF;
                /INCREMENTAL:NO
            >
            /DEBUG;
            /SUBSYSTEM:CONSOLE
        )
    endif()
endif()

################################################################################
# Dependencies
################################################################################
#if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
if(WIN32 AND MSVC)
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
         set(ADDITIONAL_LIBRARY_DEPENDENCIES
             "$<$<CONFIG:Release>:"
             "netcdf"
             ">"
             "netcdf-cxx4"
             )
    elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
         set(ADDITIONAL_LIBRARY_DEPENDENCIES
             "$<$<CONFIG:Release>:"
             "netcdf-cxx4"
             ">"
           )
    endif()
    target_link_libraries(${PROJECT_NAME} PRIVATE "${ADDITIONAL_LIBRARY_DEPENDENCIES}")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    LINK_DIRECTORIES(${NETCDF_LIBRARY_DIR} )
    MESSAGE(STATUS " add lib dir icc ${NETCDF_LIBRARY_DIR}")
    target_link_libraries(${PROJECT_NAME}  ${NETCDF_CXX_LIB} ${NETCDF_LIB})
    message(status " add netcdf lib for icc")
endif()

if(WIN32 AND MSVC)
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        target_link_directories(${PROJECT_NAME} PRIVATE
        "C:/Users/qshu/devtools/netcdflib"
        )
    elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
        target_link_directories(${PROJECT_NAME} PRIVATE
        "$<$<CONFIG:Release>:"
            "C:/Users/qshu/devtools/netcdf-cxx4-4.3.1/netcdf-cxx4-4.3.1-build/cxx4/Release"
        ">"
        )
    endif()
endif()


# the compiler flags for compiling C sources 
MESSAGE( STATUS "CMAKE_C_FLAGS: " ${CMAKE_C_FLAGS} )

# the compiler flags for compiling C++ sources 
MESSAGE( STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS} )


