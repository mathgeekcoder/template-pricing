#pragma once
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <functional>
#include <string_view>
#include <charconv>
#include "highs/Highs.h"
#include <mutex>
#include <format>

struct Timer
{
    std::chrono::steady_clock::time_point timerStart;
    std::chrono::microseconds duration;
    bool paused;

    Timer() {
		reset();
        paused = false;
    }

    void reset() {
        duration = std::chrono::microseconds::zero();
        timerStart = std::chrono::steady_clock::now();
        paused = false;
    }

    void pause() {
        if (!paused) {
            auto timerEnd = std::chrono::steady_clock::now();
            duration += std::chrono::duration_cast<std::chrono::microseconds>(timerEnd - timerStart);
            paused = true;
        }
    }

    void start() {
        timerStart = std::chrono::steady_clock::now();
        paused = false;
    }

    double TotalSeconds()
    {
        if (!paused) {
            pause();
            start();
        }
        return duration.count() / 1000000.0;
    }
};

static std::ifstream::pos_type filesize(const char* filename) {
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return in.tellg(); 
}

static inline std::string base_name(std::string const& path) {
    return path.substr(path.find_last_of("/\\") + 1);
}

static uint32_t baseTenDigits(uint64_t x) {
    uint32_t result = 1;
    while (true) {
        if (x < 10) return result;
        else if (x < 100) return result + 1;
        else if (x < 1000) return result + 2;
        else if (x < 10000) return result + 3;
        x /= 10000U;
        result += 4;
    }
}

// Recursive variadic template to bind values to the statement.
template <class T>
void _csv_index(std::ostringstream& ss, const T& a) {
    ss << a;
}

template <class T, class ...Args>
void _csv_index(std::ostringstream& ss, const T& a, const Args&... args) {
    ss << a << ",";
    _csv_index(ss, args...);
}

template <class T, class ...Args> std::string csv_join(const T& a, const Args&... args) {
	std::ostringstream ss;
    _csv_index(ss, a, args...);
	return ss.str();
}

// faster join for strings
inline std::string join(const std::vector<std::string>& v, const std::string& delim) {
    if (v.empty())
        return std::string();

    int size = delim.size() * (v.size() - 1);
    for (int i = 0; i < v.size(); ++i) {
		size += v[i].size();
    }

    std::string s;
    s.reserve(size);
    s += v[0];

    for (int i = 1; i < v.size(); ++i) {
        s += delim;
        s += v[i];
    }
    return s;
}

// faster join for integers (stringstream isn't very fast)
inline std::string join(const std::vector<int>& v, const std::string& delim) {
    if (v.empty())
        return std::string();

    int size = delim.size() * (v.size() - 1);
    for (int i = 0; i < v.size(); ++i) {
        size += baseTenDigits(abs(v[i])) + (v[i] < 0); // number of digits + possible minus sign
    }

    std::string s;
    s.reserve(size);
    s += std::to_string(v[0]);

    for (int i = 1; i < v.size(); ++i) {
        s += delim;
        s += std::to_string(v[i]);
    }
    return s;
}

inline int64_t calc_gcd(int64_t a, int64_t b) {
    int64_t h;
    if (a < 0) a = -a;
    if (b < 0) b = -b;

    if (a == 0) return b;
    if (b == 0) return a;

    do {
        h = a % b;
        a = b;
        b = h;
    } while (b != 0);

    return a;
}

inline std::string join(const std::vector<double>& v, const std::string& delim) {
    if (v.empty())
        return std::string();

    std::string s;
    s += std::to_string(v[0]);

    for (int i = 1; i < v.size(); ++i) {
        s += delim;
        s += std::to_string(v[i]);
    }
    return s;
}

template <typename _Pr>
static std::vector<uint32_t> sorted_index(size_t size, _Pr _Pred) {
	std::vector<uint32_t> index(size);
	std::iota(index.begin(), index.end(), 0);
	std::stable_sort(index.begin(), index.end(), _Pred);
	return index;
}


class ColumnAlignOutput {
   std::vector<int> _width;
   std::vector<int> _precision;
   std::vector<std::string> _name;
   std::string _output;

   template <class T>
   void display(int index, const T& a) {
	   std::format_to(std::back_inserter(_output), "{:>{}}", a, _width[index]);
   }

   void display(int index, const char*& a) {
       std::format_to(std::back_inserter(_output), "{:>{}}", a, _width[index]);
   }

   void display(int index, const double& a) {
       if (std::isnan(a) || std::isinf(a)) {
           std::format_to(std::back_inserter(_output), "{:>{}}", "-", _width[index]);
       }
       // if decimal number is larger than width, then use scientific notation
       else if (baseTenDigits(abs(a)) + _precision[index] + 3 > _width[index]) {
           std::format_to(std::back_inserter(_output), "{:>{}.{}e}", a, _width[index], _width[index] - 8);
       }
       else {
           std::format_to(std::back_inserter(_output), "{:>{}.{}f}", a, _width[index], _precision[index]);
       }
   }

   // Recursive variadic template to bind values to the statement.
   template <class T>
   void output_index(int index, const T& a) {
       display(index, a);
       std::cout << _output << std::endl;
       _output.clear(); // clear the output string
   }

   template <class T, class ...Args>
   void output_index(int index, const T& a, const Args&... args) {
       display(index, a);
       output_index(index + 1, args...);
   }

public:
    ColumnAlignOutput() = default;
    bool show_output = true;

    void write_header() {
        if (show_output) {
            for (int i = 0; i < _name.size(); ++i) {
                std::format_to(std::back_inserter(_output), "{:>{}}", _name[i], _width[i]);
            }

            std::cout << _output << std::endl;
            _output.clear(); // clear the output string
        }
	}

    void add_column(std::string name, int width, int decimals = 0) {
        _width.push_back(width);
		_precision.push_back(decimals);
		_name.push_back(name);
    }

    template <class T, class ...Args> void output(const T& a, const Args&... args) {
        if (show_output)
            output_index(0, a, args...);
    }

	const std::vector<std::string>& names() const {
		return _name;
	}
};

template <class T>
inline void erase_selected(std::vector<T>& v, const std::vector<int>& selection)
{
    v.resize(std::distance(
        v.begin(),
        std::stable_partition(v.begin(), v.end(),
            [&selection, &v](const T& item) {
                return !std::binary_search(
                    selection.begin(),
                    selection.end(),
                    static_cast<int>(static_cast<const T*>(&item) - &v[0]));
            })));
}

static bool endsWith(const std::string& str, const std::string&& suffix)
{
    return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
}

static bool startsWith(const std::string& str, const std::string&& prefix)
{
    return str.size() >= prefix.size() && 0 == str.compare(0, prefix.size(), prefix);
}

static std::string replaceAll(std::string str, const std::string from, const std::string to)
{
    size_t start_pos = 0;

    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }

    return str;
}

static std::vector<HighsInt>::const_iterator col_begin(const Highs& highs, int col) {
    auto& lp = highs.getLp();
    return lp.a_matrix_.index_.begin() + lp.a_matrix_.start_[col];
}

static std::vector<HighsInt>::const_iterator col_end(const Highs& highs, int col) {
    auto& lp = highs.getLp();
    return lp.a_matrix_.index_.begin() + lp.a_matrix_.start_[col + 1];
}

#ifdef _WIN32

struct MemoryUsage {
    size_t peak;

    MemoryUsage() {
        snapshot();
    }

    void snapshot();
    size_t PeakSize();

    std::string Peak() {
        return bytes_to_string(PeakSize());
    }

    std::string bytes_to_string(size_t bytes) {
        static const std::vector<std::string> SIZES = { " B", " KB", " MB", " GB" };
        int index = 0;
        double size = bytes;

        while (size >= 1024 && index < SIZES.size()) {
            size /= 1024.0;
            ++index;
        }

        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2) << size << SIZES[index];
        return ss.str();
    }
};


#endif

struct std_counter
{
    struct value_type { template<typename T> value_type(const T&) { } };
    void push_back(const value_type&) { ++count; }
    size_t count = 0;
    const size_t size() const { return count; }
};


// splits a csv string into it's components, assumes known size
static std::string_view csv_trim(const char*& first, const char*& last)
{
    // trim whitespace from start
    while (first < last && std::isspace(*first))
        ++first;

    if (*first == '\"')
        ++first;

    // trim whitespace from end
    auto tmp = last - 1;

    while (tmp > first && std::isspace(*tmp))
        --tmp;

    if (*tmp == '\"')
        --tmp;

    // can be "empty" string
    return std::string_view(first, tmp - first + 1);
}


static inline std::string_view csv_split_single(const std::string_view str)
{
    auto first = str.data(), last = str.data() + str.size();
    return csv_trim(first, last);
}

template <int size, char token = ','>
std::array<std::string_view, size> csv_split(const std::string_view str)
{
    std::array<std::string_view, size> output;
    bool in_quotes = false;
    int count = 0;

    for (auto first = str.data(), second = str.data(), last = str.data() + str.size(); second != last && first != last; first = second + 1) {
        // find next token outside quotes
        for (second = first; second < last; ++second) {
            if (*second == '\"') {
                in_quotes = !in_quotes;
            }
            else if (in_quotes == false && *second == token)
                break;
        }

        output[count++] = csv_trim(first, second);
    }

    return output;
}

static std::vector<std::string_view> csv_split(const std::string_view str)
{
    std::vector<std::string_view> output;
    bool in_quotes = false;

    for (auto first = str.data(), second = str.data(), last = str.data() + str.size(); second != last && first != last; first = second + 1) {
        // find next , outside quotes
        for (second = first; second < last; ++second) {
            if (*second == '\"') {
                in_quotes = !in_quotes;
            }
            else if (in_quotes == false && *second == ',')
                break;
        }

        output.emplace_back(csv_trim(first, second));
    }

    return output;
}

static void read_csv_file(std::string& filename, bool skipHeader, std::function<void(const std::string&)> processLine) {
    std::ifstream file(filename);

    if (file.is_open() == false)
        throw std::runtime_error("File not found");

    const int CHUNK = 8192;
    char out[CHUNK];
    std::string output;

    bool in_quotes = false;

    while (file.eof() == false) {
        file.read(out, CHUNK);
        std::streamsize have = file.gcount();

        char* line = (char*)out;
        bool skipHeader = true;

        while (have > 0) {
            // find end of line
            int length = 0;

            // support newlines in quotes
            while (length < have) {
                if (line[length] == '\"')
                    in_quotes = !in_quotes;
                else if (in_quotes == false && line[length] == '\n')
                    break;

                ++length;
            }

            output.insert(output.end(), line, line + length);

            if (length < have) {
                if (output.empty() == false) {
                    if (skipHeader) {
						skipHeader = false;
					}
                    else {
						processLine(output);
					}
				}

                output.clear();

                ++length;
                have -= length;
                line += length;
            }
            else {
                have = 0;
            }
        }
    }

    // last line of file didn't have a newline
    if (output.empty() == false)
        processLine(output);
}

template <typename Container, typename T, typename Lookup>
T sum(const Container& container, T init, Lookup& op) {
    return std::accumulate(container.begin(), container.end(), init, [&](T acc, const auto& elem) {
        return acc + op[elem];
    });
}


// Support for Ctrl-C signal handling
struct HandleCtrlC {
    static std::vector<std::function<void()>*> handlers; // static vector to store handlers
    static std::mutex handlers_mutex;                    // mutex for thread-safety

    HandleCtrlC(std::function<void()> cb) {
		callback = cb; // store the callback
        std::lock_guard<std::mutex> lock(handlers_mutex);
        handlers.push_back(&callback);
    }

    ~HandleCtrlC() {
        std::lock_guard<std::mutex> lock(handlers_mutex);
        handlers.erase(std::remove(handlers.begin(), handlers.end(), &callback), handlers.end());
    }

    static void call_all_handlers() {
        std::lock_guard<std::mutex> lock(handlers_mutex);
        for (const auto& handler : handlers) {
            (*handler)();
        }
    }

    static bool Enable();
    static void Disable();

private:
    std::function<void()> callback; // instance-specific callback
};

