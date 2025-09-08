#pragma once

#include <vector>
#include <iostream>
#include <functional>

#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h>

#include "fft.h"
#include "simfile.h"
#include "wave.h"

namespace nine_or_null {
    float do_the_thing();

    constexpr size_t _MAX_IMAGE_DATA_SIZE = 1000000;
    using FrequencyAxis = std::vector<float>;
    using LocalResponse = std::vector<float>;

    uint32_t heatmap(float v);

    struct StackedLocalResponse {
        // Configuration parameters
        float t_w1s;
        float t_step;
        size_t window_p2;
        size_t reduce_rate;
        std::function<float(const Signal&, const FrequencyAxis&)> frequency_filter;
        std::function<void(LocalResponse&, const LocalResponse&)> time_filter;
        
        // Derived, cached
        uint32_t sampling_rate;
        int s_w1s;
        int s_step;
        Window fft_window;
        FrequencyAxis taps;

        // Calculated
        std::vector<LocalResponse> slr;
        std::shared_ptr<float[]> value_data;
        std::shared_ptr<char[]> image_data;
        bool ready = false;

        StackedLocalResponse() :
            t_w1s(0.020f),
            t_step(0.0001f),
            window_p2(4),
            reduce_rate(2),
            value_data(new float[_MAX_IMAGE_DATA_SIZE], [](float* p){ delete[] p; }),
            image_data(new char[_MAX_IMAGE_DATA_SIZE * 4], [](char* p){ delete[] p; }) {

            frequency_filter = [](const Signal& y, const FrequencyAxis& x) {
                float acc = 0.0f;
                for (int i = 0; i < x.size(); ++i) {
                    acc += std::norm(y[i]) * x[i] * std::expf(-x[i] / 3000.0f);
                }
                return acc;
            };

            time_filter = [](LocalResponse& filtered, const LocalResponse& raw) {
                std::vector<float> kernel_rev{
                    -1.0f, 
                    -3.0f, 
                    0.0f, 
                    3.0f, 
                    1.0f
                };
                auto filtered_length = raw.size() - 2 * kernel_rev.size() + 2;
                filtered.clear();
                filtered.reserve(filtered_length);
                for (int i = 0; i <= filtered_length; ++i) {
                    float acc = 0.0f;
                    for (int j = 0; j < kernel_rev.size(); ++j) {
                        acc += raw[i+j] * kernel_rev[j];
                    }
                    filtered.push_back(acc);
                }
            };
        }

        void metadata(const Wave& wave) {
            sampling_rate = wave.wave_fmt_chunk().nSamplesPerSec;
        
            s_step = int(abs(t_step) * sampling_rate + 0.5);
            if (s_step == 0) {s_step = 1;}
            s_w1s = int(abs(t_w1s) * sampling_rate + 0.5);
            
            hann(fft_window, window_p2);
            
            taps.clear();
            taps.reserve(fft_window.size());
            for (int i = 0; i <= (1 << window_p2); ++i) {
                taps.push_back(tap(
                    i,
                    fft_window.size(),
                    wave.wave_fmt_chunk().nSamplesPerSec,
                    reduce_rate
                ));
            }
        }

        void calculate_local_response(
            LocalResponse& local_response,
            const Signal& data,
            size_t center_index
        ) {
            int n_ffts = 2 * s_w1s / s_step + 1;
            LocalResponse raw_response(n_ffts);
            Signal storage;
            for (int i = 0; i < n_ffts; ++i) {
                int index = center_index - s_w1s + i * s_step;
                fft(
                    storage,
                    data,
                    fft_window,
                    index,
                    reduce_rate
                );
                raw_response[i] = frequency_filter(storage, taps);
            }
            time_filter(local_response, raw_response);
        }

        void stack_local_response(
            const WaveData& data,
            const Simfile& simfile
        ) {
            Signal src;
            src.reserve(data.size());
            for (auto d : data) {
                src.emplace_back(d);
            }
            
            slr.clear();
            for (auto t : simfile.get_beat_times()) {
                LocalResponse local_response;
                calculate_local_response(
                    local_response,
                    src,
                    size_t(sampling_rate * t)
                );
                slr.push_back(local_response);
            }
        }

        bool update_data() {
            ready = false;

            size_t width = slr[0].size();
            size_t height = slr.size();

            for (size_t i = 0; i < _MAX_IMAGE_DATA_SIZE; ++i) {
                value_data[i] = 0;
            }
            for (size_t i = 0; i < _MAX_IMAGE_DATA_SIZE * 4; ++i) {
                image_data[i] = 0;
            }

            float max_data = 1e-12;
            for (size_t i = 0; i < height; ++i) {
                for (size_t j = 0; j < width; ++j) {
                    size_t pixel_index = i * width + j;
                    auto pixel_data = slr[i][j];
                    value_data[pixel_index] = pixel_data;
                    max_data = (max_data > pixel_data) ? max_data : pixel_data;
                }
            }
            for (size_t i = 0; i < height; ++i) {
                for (size_t j = 0; j < width; ++j) {
                    size_t pixel_index = i * width + j;
                    uint32_t pixel = heatmap(slr[i][j] / max_data);
                    for (size_t k = 0; k < 4; ++k) {
                        image_data[4*pixel_index + k] = (pixel >> (8 * k)) & 0xFF;
                    }
                }
            }

            ready = true;
            return ready;
        }

        bool update_texture(
            GLuint texture
        ) { 
            if (!ready) {
                return false;
            }

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
                slr[0].size(),
                slr.size(),
                0,
                GL_RGBA, 
                GL_UNSIGNED_BYTE,
                image_data.get()
            );

            return true;
        }
    };
}
