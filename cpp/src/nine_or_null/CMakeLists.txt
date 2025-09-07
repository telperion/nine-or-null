add_library(nine_or_null STATIC nine_or_null.cpp wave.cpp fft.cpp simfile.cpp)

find_package(OpenGL REQUIRED)

target_link_libraries(nine_or_null OpenGL::GL glfw)
