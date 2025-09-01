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
    GLuint &texture
) {
    // Create an OpenGL texture identifier
    glGenTextures(1, &texture);

    return true;
}

bool update_texture(
    GLuint texture,
    const Spectrogram &gram
) {
    // Create an OpenGL texture identifier
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
        gram.width, 
        gram.height,
        0,
        GL_RGBA, 
        GL_UNSIGNED_BYTE,
        gram.image_data.get()
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
    size_t window_size,
    size_t stride,
    size_t reduce_rate
 ) {
    nine_or_null::Signal signal;
    signal.reserve(data.size());
    for (auto d : data) {
        signal.push_back(d);
    }

    nine_or_null::Window window;
    nine_or_null::hann(window, window_size);
    size_t window_half = (window.size() - 1) / 2;

    Spectrogram gram(
        (signal.size() - window.size()) / stride,
        window_half + 1,
        4   // RGBA
    );

    size_t j = 0;
    float max_data = 1e-12;
    for (size_t center_index = 0; center_index < signal.size(); center_index += stride, ++j) {
        if (j >= gram.width) {
            break;
        }
        nine_or_null::Signal dst;
        nine_or_null::fft(
            dst,
            signal,
            window,
            center_index,
            reduce_rate
        );
        // dst.clear();
        // dst.reserve(window_half * 2 - 1);
        // for (int i = center_index - window_half; i <= center_index + window_half; ++i) {
        //     dst.push_back(signal[center_index]);
        // }

        for (size_t i = 0; i < gram.height; ++i) {
            size_t pixel_index = i * gram.width + j;
            float pixel_data = std::sqrtf(norm_squared(dst[i]));
            gram.data[pixel_index] = pixel_data;
            max_data = (max_data > pixel_data) ? max_data : pixel_data;
        }
    }

    for (size_t j = 0; j < gram.width; ++j) {
        for (size_t i = 0; i < gram.height; ++i) {
            size_t pixel_index = i * gram.width + j;
            uint32_t pixel = heatmap(gram.data[pixel_index] / max_data);

            for (size_t k = 0; k < 4; ++k) {
                gram.image_data[4*pixel_index + k] = (pixel >> (8 * k)) & 0xFF;
            }
        }
    }

    return gram;
}