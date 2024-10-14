cmake_minimum_required(VERSION 3.26)
project(nine_or_null)

set(CMAKE_CXX_STANDARD 17)

set(FETCHCONTENT_BASE_DIR "src/include")
include(FetchContent)

FetchContent_Declare(
  imgui
  GIT_REPOSITORY    https://github.com/ocornut/imgui.git
  GIT_TAG           cb16568fca5297512ff6a8f3b877f461c4323fbe
)
FetchContent_MakeAvailable(imgui)

add_subdirectory(src)
