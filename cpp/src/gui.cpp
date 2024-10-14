#include <iostream>

#include <imgui/imgui.h>
#include <imgui/imgui_impl_win32.h>
#include <imgui/imgui_impl_dx12.h>

#include <glog/logging.h>

#include "nine_or_null/nine_or_null.h"

int main(int argc, const char* argv[]) {
    std::cout << "Hello World? (GUI) " << nine_or_null::do_the_thing();
    google::InitGoogleLogging(argv[0]);
    LOG(INFO) << "Hello World! (GUI) " << nine_or_null::do_the_thing();
    
    IMGUI_CHECKVERSION();

    return 0;
}