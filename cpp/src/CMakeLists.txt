add_subdirectory(nine_or_null)

add_executable(cli cli.cpp)
add_executable(gui gui.cpp)

target_include_directories(cli PRIVATE include)
target_link_libraries(cli gflags::gflags glog::glog gtest nine_or_null)
target_include_directories(gui PRIVATE include)
target_link_libraries(gui gflags::gflags glog::glog gtest imgui nine_or_null)

add_dependencies(cli gflags::gflags glog::glog gtest nine_or_null)
add_dependencies(gui gflags::gflags glog::glog gtest imgui nine_or_null)
