#pragma once

#include <memory>
#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h>

#include "nine_or_null/nine_or_null.h"


struct Spectrogram {
    std::shared_ptr<float[]> data;
    std::shared_ptr<char[]> image_data;
    size_t width;
    size_t height;
    size_t depth;

    Spectrogram(size_t w, size_t h, size_t d) :
        width(w),
        height(h),
        depth(d),
        data(new float[w*h]),
        image_data(new char[w*h*d])
    {}

    size_t data_size() {
        return width * height;
    }
    size_t image_size() {
        return width * height * depth;
    }
};

bool prepare_texture(GLuint &texture);

bool update_texture(
    GLuint texture,
    const Spectrogram &gram
);

Spectrogram create_spectrogram(
    const nine_or_null::WaveData &data,
    size_t window_size,
    size_t stride,
    size_t reduce_rate
);

