#pragma once

#include <memory>
#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h>

#include "nine_or_null/nine_or_null.h"


struct Spectrogram {
    std::shared_ptr<float[]> data;
    std::shared_ptr<char[]> image_data;
    int width;
    int height;
    int depth;

    Spectrogram(int w, int h, int d) :
        width(w),
        height(h),
        depth(d),
        data(new float[w*h]),
        image_data(new char[w*h*d])
    {}

    int data_size() {
        return width * height;
    }
    int image_size() {
        return width * height * depth;
    }
};

bool prepare_texture(
    GLuint &texture,
    std::shared_ptr<char[]> data,
    size_t data_size,
    int width,
    int height
);

Spectrogram create_spectrogram(
    const nine_or_null::WaveData &data,
    int window_size,
    int stride,
    float scale
);

