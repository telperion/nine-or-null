#include <iostream>
#include <filesystem>

#include "nine_or_null/nine_or_null.h"

float norm_squared(nine_or_null::CC v) {
    return (v.real() * v.real() + v.imag() * v.imag());
}

int main(int argc, const char* argv[]) {
    std::filesystem::path path(argv[0]);
    std::cerr << path.parent_path().string() << std::endl;
    std::cerr << "Hello World! (CLI) " << int(nine_or_null::do_the_thing() * 86400) << std::endl;

    float sample_rate = 44100;
    float hidden_freq = 3000;
    int test_size = 6;
    nine_or_null::Window test_window;
    nine_or_null::Signal test_signal;
    nine_or_null::Signal test_fft_result;

    nine_or_null::hann(test_window, test_size);
    // test_window.assign((2 << test_size) + 1, 1.0f);
    int test_length = test_window.size();

    for (int i = 0; i < test_length; ++i) {
        test_signal.push_back(std::sinf(i * hidden_freq * nine_or_null::TAU / sample_rate));
    }

    nine_or_null::fft(
        test_fft_result,
        test_signal,
        test_window,
        1 << test_size
    );


    int max_fft_index = 0;
    float max_fft = 0.0;
    int graph_height = 40;
    for (int i = 0; i < test_fft_result.size() / 2; ++i) {
        auto v = test_fft_result[i];
        float vv = norm_squared(v);
        if (vv > max_fft) {
            max_fft_index = i;
            max_fft = vv;
        }
    }

    char buf[10];
    for (int i = 0; i < test_fft_result.size(); ++i) {
        float cr = norm_squared(test_fft_result[i]);
        int lower = (cr < 0) ? int(graph_height * cr / max_fft) + graph_height : graph_height;
        int upper = (cr > 0) ? int(graph_height * cr / max_fft) + graph_height : graph_height;

        // float cr = test_signal[i].real();
        // int lower = (cr < 0) ? int(graph_height * cr) + graph_height : graph_height;
        // int upper = (cr > 0) ? int(graph_height * cr) + graph_height : graph_height;

        std::snprintf(buf, sizeof(buf), "%4d ", i);
        std::cerr << buf;
        for (int i = 0; i < lower; ++i) {
            std::cerr << "-";
        }
        for (int i = lower; i < upper; ++i) {
            std::cerr << "#";
        }
        for (int i = upper; i < 80; ++i) {
            std::cerr << "-";
        }
        std::cerr << std::endl;
    }

    float principal_freq = (max_fft_index * sample_rate) / (test_length - 1.0f);
    std::cerr << "Max FFT @ " << principal_freq;

    return 0;
}