#include <cmath>
#include <memory>
#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h>

#include "spectrogram.h"
#include "nine_or_null/fft.h"
#include "nine_or_null/wave.h"


float norm_squared(nine_or_null::CC v) {
    return (v.real() * v.real() + v.imag() * v.imag());
}

bool prepare_texture(
    GLuint &texture,
    std::shared_ptr<char[]> data,
    size_t data_size,
    int width,
    int height
) {
    // Create an OpenGL texture identifier
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    
    // Set up filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Upload pixels into texture
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glTexImage2D(
        GL_TEXTURE_2D, 
        0, 
        GL_RGBA, 
        width, 
        height,
        0,
        GL_RGBA, 
        GL_UNSIGNED_BYTE,
        data.get()
    );

    return true;
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


Spectrogram create_spectrogram(
    const nine_or_null::WaveData &data,
    int window_size,
    int stride,
    float scale
 ) {
    nine_or_null::Signal signal;
    signal.reserve(data.size());
    for (auto d : data) {
        signal.push_back(d);
    }

    nine_or_null::Window window;
    nine_or_null::hann(window, window_size);
    int window_half = (window.size() - 1) / 2;

    Spectrogram gram(
        (signal.size() - window.size()) / stride,
        window_half + 1,
        4   // RGBA
    );

    int j = 0;
    float max_data = 1e-12;
    for (int center_index = window_half; center_index < signal.size() - window_half; center_index += stride, ++j) {
        if (j >= gram.width) {
            break;
        }
        nine_or_null::Signal dst;
        nine_or_null::fft(
            dst,
            signal,
            window,
            center_index
        );
        // dst.clear();
        // dst.reserve(window_half * 2 - 1);
        // for (int i = center_index - window_half; i <= center_index + window_half; ++i) {
        //     dst.push_back(signal[center_index]);
        // }

        for (int i = 0; i < gram.height; ++i) {
            size_t pixel_index = i * gram.width + j;
            float pixel_data = std::sqrtf(norm_squared(dst[i]));
            gram.data[pixel_index] = pixel_data;
            max_data = (max_data > pixel_data) ? max_data : pixel_data;
        }
    }

    for (int j = 0; j < gram.width; ++j) {
        for (int i = 0; i < gram.height; ++i) {
            size_t pixel_index = i * gram.width + j;
            uint32_t pixel = heatmap(gram.data[pixel_index] / max_data);

            for (int k = 0; k < 4; ++k) {
                gram.image_data[4*pixel_index + k] = (pixel >> (8 * k)) & 0xFF;
            }
        }
    }

    return gram;
}