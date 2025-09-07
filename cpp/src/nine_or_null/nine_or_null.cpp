#include <ctime>

#include "nine_or_null.h"


namespace nine_or_null {
    float do_the_thing() {
        std::time_t time_raw;
        struct std::tm * time_ptr;

        std::time(&time_raw);
        time_ptr = std::gmtime(&time_raw);

        return (time_ptr->tm_hour * 3600.0f + time_ptr->tm_min * 60.0f + time_ptr->tm_sec) / 86400.0f;
    }

    uint32_t heatmap(float v) {
        // ABGR order
        if (v < 0) {
            return 0xFF000000;
        }
        if (v < 0.1) {
            return 0xFF000000 + int(0xFF * v / 0.1);
        }
        if (v < 0.4) {
            return 0xFF0000FF + (int(0xFF * (v - 0.1) / 0.3) << 8);
        }
        if (v < 1.0) {
            return 0xFF00FFFF + (int(0xFF * (v - 0.4) / 0.6) << 16);
        }
        return 0xFFFFFFFF;
    }
}
