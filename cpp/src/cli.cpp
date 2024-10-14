#include <iostream>

#include <glog/logging.h>

#include "nine_or_null/nine_or_null.h"

int main(int argc, const char* argv[]) {
    std::cout << "Hello World? (CLI) " << int(nine_or_null::do_the_thing() * 86400) << std::endl;
    //google::InitGoogleLogging(argv[0]);
    //LOG(INFO) << "Hello World! (CLI) " << int(nine_or_null::do_the_thing() * 86400);

    return 0;
}