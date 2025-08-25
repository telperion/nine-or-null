#pragma once

#include <cmath>

#include <vector>
#include <complex>

namespace nine_or_null {
    using CC = std::complex<float>;
    using Window = std::vector<float>;
    using Signal = std::vector<CC>;

    const double PI = 4.0f * std::atanf(1.0f);
    const double TAU = 2.0f * PI;

    float do_the_thing();
    
    /**
    @brief Generate a Hann window for FFT use.

    @param window Vector storage for the window.
    @param size One-sided length of the Hann window, expressed as a power on 2.
    */
    void hann(Window &window, int size);

    /**
    @brief Perform an FFT (or IFFT) with the given window.

    @param dst Storage for the result of the transform.
    @param src Input signal.
    @param window Filtering window (e.g. the Hann function).
    @param center_index The index into the input signal that represents the center of the window currently being analyzed.
    */
    void fft(
        Signal &dst,
        const Signal &src,
        const Window &window,
        int center_index,
        bool invert = false
    );
}
