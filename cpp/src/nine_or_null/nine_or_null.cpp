#include <cmath>
#include <complex>
#include <ctime>
#include <iostream>
#include <vector>

#include "nine_or_null.h"

namespace nine_or_null {
    float do_the_thing() {
        std::time_t time_raw;
        struct std::tm * time_ptr;

        std::time(&time_raw);
        time_ptr = std::gmtime(&time_raw);

        return (time_ptr->tm_hour * 3600.0f + time_ptr->tm_min * 60.0f + time_ptr->tm_sec) / 86400.0f;
    }

    void hann(Window &window, int size) {
        int length = (2 << size);
        window.reserve(length + 1);
        for (int i = 0; i <= length; ++i) {
            window.push_back((1.0f - std::cosf((TAU * i) / length)) * 0.5f);
        }
    }

    int bit_reversal(int a, int bits) {
        int b = 0;
        for (int i = 0; i < bits; i++) {
            b <<= 1;
            b += (a & 1);
            a >>= 1;
        }
        return b;
    }

    void bit_reverser(
        std::vector<int> &br,
        int window_length
    ) {
        int window_log2 = 0;
        while ((1 << window_log2) < window_length) {
            ++window_log2;
        }

        // Pre-arrange the order for the butterfly transform to pick up the signal values.
        br.clear();
        for (int i = 0; i < window_length; i++) {
            br.push_back(bit_reversal(i, window_log2));
        }
        br.push_back(window_length);
    }

    void fft(
        Signal &dst,
        const Signal &src,
        const Window &window,
        int center_index,
        bool invert
    ) {
        // How many steps of butterfly transform to perform?
        int window_length = window.size() - 1;
        int window_offset = center_index - window_length / 2;

        // Set up the butterfly transform using stride lengths.
        std::vector<int> reorder(window_length + 1);
        bit_reverser(reorder, window_length);
        for (const auto b : reorder) {
            std::cerr << b << " ";
        }
        std::cerr << std::endl;

        dst.clear();
        dst.reserve(window_length + 1);
        for (int i = 0; i <= window_length; ++i) {
            dst.push_back(src[reorder[i] + window_offset] * window[reorder[i]]);
        }

        for (int stride = 2; stride <= window_length; stride <<= 1) {
            int half_stride = stride / 2;
            float theta = (TAU / stride) * (invert ? -1 : 1);
            CC unity(std::cosf(theta), std::sinf(theta));
            for (int i = 0; i < window_length; i += stride) {
                CC winding(1);
                for (int j = 0; j < half_stride; ++j) {
                    int butter_index = i + j;
                    int fly_index = i + j + half_stride;
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
                x /= window_length;
            }
        }
    }
}
