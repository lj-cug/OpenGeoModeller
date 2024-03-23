SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
# add a exe
ADD_EXECUTABLE(helloSLAM helloSLAM.cpp)

# add a lib
add_library( hello STATIC libHelloSLAM.cpp )
# add a share lib
add_library( hello_shared SHARED libHelloSLAM.cpp )
SET_TARGET_PROPERTIES(hello_shared PROPERTIES OUTPUT_NAME "hello")
GET_TARGET_PROPERTY(OUTPUT_VALUE hello_shared OUTPUT_NAME)
MESSAGE(STATUS "This is the hello_shared OUTPUT_NAME:"${OUTPUT_VALUE})

add_executable( useHello useHello.cpp )
set(CMAKE_INSTALL_PREFIX "/home/qy/2018-11-20-install/1119exercise")
set_target_properties(useHello PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
#SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
message(STATUS "${CMAKE_INSTALL_PREFIX}")
#message(STATUS "${INSTALL_RPATH}")
#message(STATUS "${CMAKE_INSTALL_RPATH}")
# 将库文件链接到可执行程序上
target_link_libraries( useHello hello_shared )
INSTALL(TARGETS useHello hello_shared hello
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION libstatic
    )
INSTALL(FILES ../include/libHelloSLAM.h DESTINATION include)
