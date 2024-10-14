add_subdirectory(nine_or_null)

add_executable(cli cli.cpp)
add_executable(gui gui.cpp)

target_include_directories(cli PRIVATE ${glog_SOURCE_DIR})
target_link_libraries(cli glog::glog nine_or_null)
target_include_directories(gui PRIVATE ${glog_SOURCE_DIR} ${imgui_SOURCE_DIR})
target_link_libraries(gui glog::glog nine_or_null)

add_dependencies(cli glog::glog nine_or_null)
add_dependencies(gui glog::glog nine_or_null)

# add_library(imgui
# 	${imgui_SOURCE_DIR}/imgui.cpp
# 	${imgui_SOURCE_DIR}/imgui_draw.cpp
# 	${imgui_SOURCE_DIR}/imgui_tables.cpp
# 	${imgui_SOURCE_DIR}/imgui_widgets.cpp
# 	${imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp
# 	${imgui_SOURCE_DIR}/backends/imgui_impl_dx12.cpp
# )
target_include_directories(gui PUBLIC ${imgui_SOURCE_DIR})
target_sources(
	gui
	PRIVATE
	${imgui_SOURCE_DIR}/imgui.cpp
	${imgui_SOURCE_DIR}/imgui_demo.cpp
	${imgui_SOURCE_DIR}/imgui_draw.cpp
	${imgui_SOURCE_DIR}/imgui_tables.cpp
	${imgui_SOURCE_DIR}/imgui_widgets.cpp
	${imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp
	${imgui_SOURCE_DIR}/backends/imgui_impl_dx11.cpp
)
