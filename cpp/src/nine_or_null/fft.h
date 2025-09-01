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
    
    /**
    @brief Generate a Hann window for FFT use.

    @param window Vector storage for the window.
    @param size One-sided length of the Hann window, expressed as a power on 2.
    */
    void hann(Window &window, size_t size);

    /**
    @brief Perform an FFT (or IFFT) with the given window.

    @param dst Storage for the result of the transform.
    @param src Input signal.
    @param window Filtering window (e.g. the Hann function).
    @param center_index The index into the input signal that represents the center of the window currently being analyzed.
    @param reduce_rate Reduce effective sampling rate by spacing out samples pulled from the input signal by this multiplier.
    @param invert IFFT, instead of FFT.
    */
    void fft(
        Signal &dst,
        const Signal &src,
        const Window &window,
        size_t center_index,
        size_t reduce_rate = 1,
        bool invert = false
    );

    /**
    @brief Calculate the frequency tap for the given index of an FFT calculated with the given parameters.

    @param index The index into the FFT.
    @param window_length Filtering window length (window.size()).
    @param sampling_rate Samples per second (Hz).
    @param reduce_rate If the sample rate was artificially decreased using this parameter, also provide it here.
    @return The frequency tap at the given index.
    */
    float tap(
        size_t index,
        size_t window_length,
        float sampling_rate,
        size_t reduce_rate = 1
    );
}
