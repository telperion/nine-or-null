#include <iostream>

#include <glog/logging.h>

#include "nine_or_null/nine_or_null.h"

int main(int argc, const char* argv[]) {
    std::cout << "Hello World? (CLI) " << nine_or_null::do_the_thing();
    google::SetLogDestination(google::GLOG_INFO, ".");
    google::InitGoogleLogging(argv[0]);
    LOG(INFO) << "Hello World! (CLI) " << nine_or_null::do_the_thing();

    return 0;
}