cmake_minimum_required(VERSION 3.26)
project(nine_or_null)

set(CMAKE_CXX_STANDARD 17)

set(FETCHCONTENT_BASE_DIR "src/include")
include(FetchContent)

FetchContent_Declare(
  gflags
  GIT_REPOSITORY    https://github.com/gflags/gflags.git
  GIT_TAG           e171aa2d15ed9eb17054558e0b3a6a413bb01067
)
FetchContent_MakeAvailable(gflags)

FetchContent_Declare(
  imgui
  GIT_REPOSITORY    https://github.com/ocornut/imgui.git
  GIT_TAG           cb16568fca5297512ff6a8f3b877f461c4323fbe
)
FetchContent_Declare(
  gtest
  GIT_REPOSITORY    https://github.com/google/googletest.git
  GIT_TAG           b514bdc898e2951020cbdca1304b75f5950d1f59
)
FetchContent_Declare(
  glog
  GIT_REPOSITORY    https://github.com/google/glog.git
  GIT_TAG           7b134a5c82c0c0b5698bb6bf7a835b230c5638e4
)
FetchContent_MakeAvailable(glog gtest imgui)

add_subdirectory(src)
