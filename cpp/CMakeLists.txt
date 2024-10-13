cmake_minimum_required(VERSION 3.26)
project(nine_or_null)

add_subdirectory(src)
target_link_libraries(cli nine_or_null)
target_link_libraries(gui nine_or_null)

# find_package (Threads REQUIRED) #find pthreads package
# target_link_libraries (my_exe my_lib ${CMAKE_THREAD_LIBS_INIT}) #link against pthreads and my_lib