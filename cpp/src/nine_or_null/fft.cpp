#include <cmath>
#include <complex>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "fft.h"

namespace nine_or_null {
    namespace {
        // Hide in an anonymous namespace
        template<typename T>
        const T& at_or(const std::vector<T> &v, size_t i, const T& d) {
            if (i < 0) {
                return d;
            }
            if (i >= v.size()) {
                return d;
            }
            return v[i];
        }
    }

    void hann(Window &window, size_t size) {
        size_t length = (2 << size);
        window.reserve(length + 1);
        for (size_t i = 0; i <= length; ++i) {
            window.push_back((1.0f - std::cosf((TAU * i) / length)) * 0.5f);
        }
    }

    size_t bit_reversal(size_t a, size_t bits) {
        size_t b = 0;
        for (size_t i = 0; i < bits; i++) {
            b <<= 1;
            b += (a & 1);
            a >>= 1;
        }
        return b;
    }

    void fft(
        Signal &dst,
        const Signal &src,
        const Window &window,
        size_t center_index,
        size_t reduce_rate,
        bool invert
    ) {
        // How many steps of butterfly transform to perform?
        auto dst_window_length = window.size() - 1;
        auto src_window_length = dst_window_length * reduce_rate;
        auto window_offset = center_index - src_window_length / 2;

        // Prepare the elements in the window for the transform.
        int window_log2 = 0;
        while ((1 << window_log2) < dst_window_length) {
            ++window_log2;
        }

        dst.clear();
        dst.reserve(dst_window_length + 1);
        for (size_t i = 0; i < dst_window_length; ++i) {
            size_t bit_reversed = bit_reversal(i, window_log2);
            dst.push_back(at_or(
                src, 
                bit_reversed * reduce_rate + window_offset, 
                CC(0.0f)
            ) * window[bit_reversed]);
        }
        dst.push_back(at_or(
            src,
            src_window_length + window_offset,
            CC(0.0f)
        ) * window[dst_window_length]);

        // Set up the butterfly transform using stride lengths.
        for (size_t stride = 2; stride <= dst_window_length; stride <<= 1) {
            size_t half_stride = stride / 2;
            float theta = (TAU / stride) * (invert ? -1 : 1);
            CC unity(std::cosf(theta), std::sinf(theta));
            for (size_t i = 0; i < dst_window_length; i += stride) {
                CC winding(1);
                for (int j = 0; j < half_stride; ++j) {
                    size_t butter_index = i + j;
                    size_t fly_index = i + j + half_stride;
                    CC butter(dst[butter_index]);
                    CC fly(dst[fly_index] * winding);
                    dst[butter_index] = butter + fly;
                    dst[fly_index] = butter - fly;
                    winding *= unity;
                }
            }
        }

        // Scaling when applying the IFFT.
        if (invert) {
            for (CC &x : dst) {
                x /= dst_window_length;
            }
        }
    }

    float tap(
        size_t index,
        size_t window_length,
        float sampling_rate,
        size_t reduce_rate
    ) {
        float delta_f = sampling_rate / (2.0f * reduce_rate * window_length);
        if (index < (window_length+1)/2) {
            return index * delta_f;
        }
        return (window_length - index) * delta_f;
    }
}
