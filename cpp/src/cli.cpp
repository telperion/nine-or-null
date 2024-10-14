#include <iostream>
#include <filesystem>

#include "nine_or_null/nine_or_null.h"

int main(int argc, const char* argv[]) {
    std::filesystem::path path(argv[0]);
    std::cerr << argv[0] << std::endl;
    std::cerr << "Hello World! (CLI) " << int(nine_or_null::do_the_thing() * 86400) << std::endl;

    return 0;
}