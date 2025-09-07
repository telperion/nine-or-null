#pragma once

#include <iostream>
#include <sstream>
#include <numeric>
#include <iomanip>
#include <vector>
#include <string>

constexpr int _MAX_QUANTIZATION = 48;
constexpr int _MAX_PRECISION = 3;
const double _DIV_PRECISION = std::pow(10, _MAX_PRECISION);


const std::string fmt_precision();
double to_precision(double v);

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
        std::vector<Event> bpms;
        std::vector<Event> stops;
        std::vector<NoteHead> notes;
};

