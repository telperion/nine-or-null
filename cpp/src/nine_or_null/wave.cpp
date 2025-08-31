#include <cstdint>
#include <cstring>
#include <ios>
#include <iomanip>
#include <iostream>
#include <memory>
#include <fstream>

#include "wave.h"

// https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html

#ifndef _WAVE_READ
#define _WAVE_READ(dst, sz) \
    is.read((dst), (sz)); \
    total_bytes_read += (sz)
#endif

namespace nine_or_null {
    std::istream& operator>>(std::istream& is, Wave& obj) {
        bool error = false;
        uint64_t total_bytes_read = 0;
        uint64_t file_size_bytes = 0;

        is.seekg(0, is.end);
        file_size_bytes = is.tellg();
        is.seekg(0, is.beg);

        WaveChunkHeader master_chunk;
        _WAVE_READ(reinterpret_cast<char*>(&master_chunk), sizeof(WaveChunkHeader));
        if (std::strncmp(master_chunk.ckID, "RIFF", 4) != 0) {
            std::cerr << "Master chunk ID: \"" << master_chunk.ckID << "\" (expected \"RIFF\")" << std::endl;
            error = true;
        }

        char wave_id[4];
        _WAVE_READ(wave_id, 4);
        if (std::strncmp(wave_id, "WAVE", 4) != 0) {
            std::cerr << "Wave ID: \"" << wave_id << "\" (expected \"WAVE\")" << std::endl;
            error = true;
        }

        bool read_format = false;
        bool read_fact = false;
        WaveChunkHeader chunk;
        _WAVE_READ(reinterpret_cast<char*>(&chunk), sizeof(WaveChunkHeader));
        while (!is.eof() && !error) {
            if (std::strncmp(chunk.ckID, "fmt ", 4) == 0) {
                _WAVE_READ(reinterpret_cast<char*>(&obj._wave_fmt_chunk), sizeof(WaveFmtChunk));
                if (chunk.cksize == 18) {
                    _WAVE_READ(reinterpret_cast<char*>(&obj._wave_fmt_chunk_ext.cbSize), 2);
                }
                else if (chunk.cksize == 40) {
                    _WAVE_READ(reinterpret_cast<char*>(&obj._wave_fmt_chunk_ext), sizeof(WaveFmtChunkExt));
                }
                else if (chunk.cksize != 16) {
                    std::cerr << "fmt chunk: " << chunk.cksize << " (expected 16, 18, or 40)" << std::endl;
                    error = true;
                }
                read_format = true;
            }
            else if (std::strncmp(chunk.ckID, "fact", 4) == 0) {
                _WAVE_READ(reinterpret_cast<char*>(&obj._dwSampleLength), 4);
                read_fact = true;
            }
            else if (std::strncmp(chunk.ckID, "data", 4) == 0) {
                if (!read_format) {
                    std::cerr << "data chunk arrived before fmt chunk" << std::endl;
                    error = true;
                    continue;
                }
                int aligned_chunk_size = chunk.cksize + (chunk.cksize % 2);
                obj._samples = std::make_unique<char[]>(aligned_chunk_size);
                obj._num_bytes_for_samples = chunk.cksize;
                _WAVE_READ(obj._samples.get(), aligned_chunk_size);
            }
            else {
                std::cout << "Unknown chunk ID: \"" << chunk.ckID << "\", size " << chunk.cksize << " (expected \"fmt \", \"fact\", or \"data\")." << std::endl;
                std::unique_ptr<char[]> throwaway = std::make_unique<char[]>(chunk.cksize + chunk.cksize % 2);
                _WAVE_READ(reinterpret_cast<char*>(throwaway.get()), chunk.cksize + chunk.cksize % 2);
            }

            _WAVE_READ(reinterpret_cast<char*>(&chunk), sizeof(WaveChunkHeader));
        }

        if (error) {
            is.setstate(std::ios::failbit);
        }
        if (!read_fact) {
            obj._dwSampleLength = 8 * obj._num_bytes_for_samples / obj._wave_fmt_chunk.wBitsPerSample / obj._wave_fmt_chunk.nChannels;
            std::cout << "No fact chunk; interpolating number of samples (per channel)" << std::endl
                      << obj._num_bytes_for_samples << " bytes / "
                      << obj._wave_fmt_chunk.wBitsPerSample / 8 << " bytes per sample / "
                      << obj._wave_fmt_chunk.nChannels << " channels = "
                      << obj._dwSampleLength << " samples (per channel)" << std::endl;
        }
        return is;
    }

    std::ostream& operator<<(std::ostream& os, const Wave& obj) {
        os  << "Wave file: " << std::endl
            << "    Format tag: " << std::hex << obj._wave_fmt_chunk.wFormatTag << std::dec << std::endl
            << "    Number of interleaved channels: " << obj._wave_fmt_chunk.nChannels << std::endl
            << "    Sampling rate (blocks per second): " << obj._wave_fmt_chunk.nSamplesPerSec << std::endl
            << "    Data rate: " << obj._wave_fmt_chunk.nAvgBytesPerSec << std::endl
            << "    Data block size (bytes): " << obj._wave_fmt_chunk.nBlockAlign << std::endl
            << "    Bits per sample: " << obj._wave_fmt_chunk.wBitsPerSample << std::endl;
        if (obj._wave_fmt_chunk_ext.cbSize != 0) {
            os  << "    Size of the extension: " << obj._wave_fmt_chunk_ext.cbSize << std::endl
                << "    Number of valid bits: " << obj._wave_fmt_chunk_ext.wValidBitsPerSample << std::endl
                << "    Speaker position mask: " << obj._wave_fmt_chunk_ext.dwChannelMask << std::endl
                << "    GUID, including the data format code: " << obj._wave_fmt_chunk_ext.SubFormat << std::endl;
        }
        os  << "    Number of samples (per channel): " << obj._dwSampleLength << std::endl
            << "    Number of bytes of sample data: " << obj._num_bytes_for_samples << std::endl
            << std::endl << std::hex;
        // for (int i = 0; i < obj._num_bytes_for_samples; i += 32) {
        //     os << std::hex << std::setfill('0') << std::setw(8) << i << " ";
        //     for (int j = 0; j < 8; j++) {
        //         os << std::hex << std::setfill('0') << std::setw(8) << *reinterpret_cast<std::uint32_t*>(obj._samples.get() + i + j*4);
        //     }
        //     os << std::endl;
        // }
        // os << std::endl;
        return os;
    }

    void Wave::fill(WaveData &dst, int channel) const {
        size_t bps = _wave_fmt_chunk.wBitsPerSample / 8;
        size_t stride = _wave_fmt_chunk.nBlockAlign;

        dst.clear();
        dst.reserve(_dwSampleLength);
        switch (_wave_fmt_chunk.wFormatTag) {
            case WaveFormat::WAVE_FORMAT_PCM:
                for (int i = 0; i < _dwSampleLength; ++i) {
                    size_t offset = i * stride + channel * bps;
                    size_t sign_align = 32 - _wave_fmt_chunk.wBitsPerSample;
                    int32_t acc = *reinterpret_cast<int32_t*>(_samples.get() + offset);
                    acc <<= sign_align;
                    acc >>= sign_align;
                    dst.push_back(float(acc));
                }
            break;
            case WaveFormat::WAVE_FORMAT_IEEE_FLOAT:
                for (int i = 0; i < _dwSampleLength; ++i) {
                    size_t offset = i * stride + channel * bps;
                    dst.push_back(*reinterpret_cast<float*>(&_samples[offset]));
                }
            break;
            default:
                std::cerr << "Format not yet digestible by wave library: "
                          << std::hex << std::setfill('0') << std::setw(2)
                          << _wave_fmt_chunk.wFormatTag << std::endl;
        }
    }

    void demo_wave(int argc, char* argv[]) {
        std::cout << "Hello, from nine_or_null::wave!\n";
        std::ifstream fp;
        fp.open(argv[1], std::ios::in | std::ios::binary);
        Wave test;
        fp >> test;
        fp.close();
        std::cout << test;
        std::cout << "Goodbye, from nine_or_null::wave!\n";
    }
}


