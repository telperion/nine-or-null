#pragma once

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

// https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html

namespace nine_or_null {
    using WaveData = std::vector<float>;

    enum WaveFormat { 
        WAVE_FORMAT_PCM = 0x0001,
        WAVE_FORMAT_IEEE_FLOAT = 0x0003,
        WAVE_FORMAT_ALAW = 0x0006,
        WAVE_FORMAT_MULAW = 0x0007,
        WAVE_FORMAT_EXTENSIBLE = 0xFFFE
    };

    #pragma pack(push, 2)
    struct WaveChunkHeader {
        char ckID[4];                           // Chunk ID
        std::uint32_t cksize;                   // Chunk size
    };

    struct WaveFmtChunk {                       // Chunk ID: "fmt "
        std::uint16_t wFormatTag;               // Format code
        std::uint16_t nChannels;                // Number of interleaved channels
        std::uint32_t nSamplesPerSec;           // Sampling rate (blocks per second)
        std::uint32_t nAvgBytesPerSec;          // Data rate
        std::uint16_t nBlockAlign;              // Data block size (bytes)
        std::uint16_t wBitsPerSample;           // Bits per sample
    };

    struct WaveFmtChunkExt {
        std::uint16_t cbSize;                   // Size of the extension (0 or 22)
        std::uint16_t wValidBitsPerSample;      // Number of valid bits
        std::uint32_t dwChannelMask;            // Speaker position mask
        char SubFormat[16];                     // GUID, including the data format code
    };
    #pragma pack(pop)

    class Wave {
        public:
            Wave() noexcept:
            _wave_fmt_chunk(),
            _wave_fmt_chunk_ext(),
            _dwSampleLength(),
            _samples(),
            _num_bytes_for_samples(0)
            {

            }

            const WaveFmtChunk& wave_fmt_chunk() const {
                return _wave_fmt_chunk;
            }

            const WaveFmtChunkExt& wave_fmt_chunk_ext() const {
                return _wave_fmt_chunk_ext;
            }

            char& operator[](int i) noexcept {
                return _samples[i];
            }
            const char& operator[](int i) const noexcept {
                return _samples[i];
            }
            auto size() const noexcept {
                return _num_bytes_for_samples;
            }

            auto dwSampleLength() const noexcept {
                return _dwSampleLength;
            }

            auto length() const {
                return float(_dwSampleLength) / float(_wave_fmt_chunk.nSamplesPerSec);
            }

            void fill(WaveData &dst, int channel) const;


        private:
            WaveFmtChunk _wave_fmt_chunk;
            WaveFmtChunkExt _wave_fmt_chunk_ext;
            std::uint32_t _dwSampleLength;
            std::unique_ptr<char[]> _samples;
            std::size_t _num_bytes_for_samples;


        friend std::istream& operator>>(std::istream& is, Wave& obj);
        friend std::ostream& operator<<(std::ostream& os, const Wave& obj);
    };

    std::istream& operator>>(std::istream& is, Wave& obj);
    std::ostream& operator<<(std::ostream& os, const Wave& obj);

    void demo_wave(int argc, char* argv[]);
}

