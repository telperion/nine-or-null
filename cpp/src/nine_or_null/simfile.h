#pragma once

#include <algorithm>
#include <iostream>
#include <sstream>
#include <numeric>
#include <iomanip>
#include <vector>
#include <string>

#define _MIN(a, b) (((a) < (b)) ? (a) : (b))
#define _MAX(a, b) (((a) > (b)) ? (a) : (b))

constexpr int _MAX_BEAT_SENTINEL = 1000000;
constexpr int _MAX_QUANTIZATION = 48;
constexpr int _MAX_PRECISION = 3;
const double _DIV_PRECISION = std::pow(10, _MAX_PRECISION);


const std::string fmt_precision();
double to_precision(double v);

using TimeAxis = std::vector<double>;

struct BeatFraction {
    int64_t n;
    uint64_t d;

    BeatFraction(double b) {
        int64_t n_mq = int64_t(b * _MAX_QUANTIZATION + 0.5);
        int64_t g = std::gcd(n_mq, _MAX_QUANTIZATION);
        g = (g < 0) ? -g : g;
        n = n_mq / g;
        d = _MAX_QUANTIZATION / g;
    }

    operator double() const {
        return to_precision(double(n) / double(d));
    }

    double exact() const {
        return double(n) / double(d);
    }

    friend std::ostream& operator<<(std::ostream& os, const BeatFraction& obj) {
        os << std::fixed << std::setprecision(_MAX_PRECISION) << double(obj);
        return os;
    }

    std::string toString() const {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }
};

struct Event {
    double beat;
    double value;

    Event(double b = 0.0, double v = 0.0) : beat(BeatFraction(b)), value(v) {}

    friend std::istream& operator>>(std::istream& is, Event& obj) {
        double b;
        char equal_sign;
        is >> std::skipws >> b >> equal_sign >> obj.value;
        obj.beat = b;
        return is;
    }

    friend std::ostream& operator<<(std::ostream& os, const Event& obj) {
        os << std::fixed << std::setprecision(_MAX_PRECISION) << double(BeatFraction(obj.beat)) << "=" << double(obj.value);
        return os;
    }

    std::string toString() const {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }

    bool operator<(const Event& o) {
        return double(beat) < double(o.beat);
    }

    double exact_beat() const {
        return BeatFraction(beat).exact();
    }
};

struct NoteHead {
    double beat;
    uint32_t lane;
};

struct Field {
    std::string tag;
    std::string contents;

    static void read_up_to(std::istream& is, std::string& ss, char sentinel) {
        do {
            char next;
            is.get(next);
            if (is.eof()) break;
            if (next == sentinel) break;
            ss += next;
        } while (true);
    }

    friend std::istream& operator>>(std::istream& is, Field& obj) {
        char leading_hash;
        obj.tag = "";
        obj.contents = "";
        is >> std::skipws >> leading_hash >> std::noskipws;
        if (is.eof()) {
            return is;
        }
        if (leading_hash != '#') {
            is.putback(leading_hash);
            return is;
        }
        read_up_to(is, obj.tag, ':');
        read_up_to(is, obj.contents, ';');
        return is;
    }
};

class Simfile {
    public:
        friend std::istream& operator>>(std::istream& is, Simfile& obj) {
            do {
                Field field;
                is >> field;
                if (is.eof()) break;
                if (field.tag == "" && field.contents == "") break;

                if (field.tag == "TITLE") {
                    obj.title = field.contents;
                }
                else if (field.tag == "ARTIST") {
                    obj.artist = field.contents;
                }
                else if (field.tag == "OFFSET") {
                    obj.offset = std::stod(field.contents);
                }
                else if (field.tag == "BPMS" && field.contents.length() > 0) {
                    std::stringstream ss(field.contents);
                    parse_event_list(ss, obj.bpms);
                }
                else if (field.tag == "STOPS" && field.contents.length() > 0) {
                    std::stringstream ss(field.contents);
                    parse_event_list(ss, obj.stops);
                }
            } while (true);
            return is;
        }

        friend std::ostream& operator<<(std::ostream& os, const Simfile& obj) {
            os << "Title: " << obj.title << std::endl
               << "Artist: " << obj.artist << std::endl
               << "BPMs:" << std::endl;
            for (auto event : obj.bpms) {
                os << "\t" << event << std::endl;
            }
            os << "Stops:" << std::endl;
            for (auto event : obj.stops) {
                os << "\t" << event << std::endl;
            }
            return os;
        }

        const TimeAxis& get_beat_times() const {
            return beat_times;
        }

        const TimeAxis& get_note_times() const {
            return note_times;
        }

        void set_dirty() {
            dirty = true;
        }

        bool is_dirty() const {
            return dirty;
        }

        void sort() {
            std::sort(bpms.begin(), bpms.end());
            std::sort(stops.begin(), stops.end());
        }

        double beat_to_time(double beat) {
            double exact_beat = BeatFraction(beat).exact();
            double last_beat = 0.0;
            double acc = offset;

            if (bpms.empty()) {
                // No BPMs at all??
                return acc;
            }
            if (bpms[0].beat > 0) {
                // First BPM occurs after beat zero??
                return acc;
            }

            // Handle BPM changes
            auto last_bpm_event = bpms[0];
            for (auto event : bpms) {
                auto event_beat = event.exact_beat();
                if (event_beat > last_beat) {
                    double next_beat = _MIN(event_beat, beat);
                    next_beat = _MAX(next_beat, 0.0);
                    
                    acc += (next_beat - last_beat) * 60.0 / last_bpm_event.value;
                    last_bpm_event = event;
                    last_beat = next_beat;
                }
                if (event_beat > exact_beat) {
                    break;
                }
            }
            if (last_bpm_event.beat < beat) {
                acc += (beat - last_bpm_event.beat) * 60.0 / last_bpm_event.value;
            }

            // Handle stops
            for (auto event : stops) {
                if (event.exact_beat() < exact_beat) {
                    // Non-inclusive (note @ beat registers)
                    acc += event.value;
                }
            }

            return acc;
        }

        void calculate_times(const double max_time) {
            beat_times.clear();
            for (int i = 0; i < _MAX_BEAT_SENTINEL; ++i) {
                auto t = beat_to_time(i);
                if (t > max_time) {
                    break;
                }
                beat_times.push_back(t);
            }
            for (auto note : notes) {
                note_times.push_back(beat_to_time(note.beat));
            }
        }

        void cleanup(const double max_time) {
            if (dirty) {
                //sort();
                calculate_times(_MIN(last_second_hint, max_time));
            }
            dirty = false;
        }

    public:
        static void parse_event_list(std::istream& is, std::vector<Event> &v) {
            char comma;
            is >> std::skipws;
            while (!is.eof()) {
                Event event;
                is >> event;
                v.push_back(event);
                is >> comma;
            }
        }

        std::string title;
        std::string artist;
        double offset;
        double last_second_hint;
        std::vector<Event> bpms;
        std::vector<Event> stops;
        std::vector<NoteHead> notes;
        
        TimeAxis beat_times;
        TimeAxis note_times;

        bool dirty = false;
};

