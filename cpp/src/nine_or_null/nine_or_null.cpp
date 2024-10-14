#include <ctime>

#include "nine_or_null.h"

namespace nine_or_null {
    float do_the_thing() {
        std::time_t time_raw;
        struct std::tm * time_ptr;

        std::time(&time_raw);
        time_ptr = std::gmtime(&time_raw);

        return (time_ptr->tm_hour * 60.0f + time_ptr->tm_min) / 1440.0f;
    }
}
