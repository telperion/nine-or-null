#include "simfile.h"

const std::string fmt_precision() {
    return std::string("%0.") + char('0' + _MAX_PRECISION) + "f";
}

double to_precision(double v) {
    return int(_DIV_PRECISION * v + 0.5) / _DIV_PRECISION;
}