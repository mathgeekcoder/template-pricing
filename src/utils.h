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
#include <span>
#include "extern/gurobi/gurobi_highs.h"
#include <filesystem>
#include <regex>
#include "zlib.h"

namespace fs = std::filesystem;

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
       else if (baseTenDigits(static_cast<uint64_t>(abs(a))) + static_cast<uint32_t>(_precision[index] + 3) > static_cast<uint32_t>(_width[index])) {
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

static std::span<const HighsInt> get_column(const Highs& highs, size_t col) {
    auto& lp = highs.getLp();
    return std::span<const HighsInt>(lp.a_matrix_.index_.cbegin() + lp.a_matrix_.start_[col], lp.a_matrix_.start_[col + 1] - lp.a_matrix_.start_[col]);
}

static bool getColsCost(const Highs& highs, size_t start, size_t end, double* costs) {
    auto& lp = highs.getLp();
    
    if (start < end && end < lp.num_col_) {
        // copy is defined on [start, end)
        std::copy(lp.col_cost_.cbegin() + start, lp.col_cost_.cbegin() + end + 1, costs);
        return true;
    }
    else {
        return false;
    }
}

#ifdef SUPPORT_GUROBI

static std::vector<HighsInt> get_column(const GurobiHighs& g, size_t col) {
    return g.get_col(col);
}

static bool getColsCost(const GurobiHighs& g, size_t start, size_t end, double* costs) {
    return g.getColsCost(start, end, costs);
}

#endif

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
        double size = static_cast<double>(bytes);

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

template <typename Container, typename T, typename Lookup>
T sum(const Container& container, T init, Lookup& op) {
    return std::accumulate(container.begin(), container.end(), init, [&](T acc, const auto& elem) {
        return acc + op[elem];
    });
}

template <typename Iterator, typename T, typename Lookup>
T sum(Iterator begin, Iterator end, T init, Lookup& op) {
    return std::accumulate(begin, end, init, [&](T acc, const auto& elem) {
        return acc + op[elem];
    });
}

// deletes all indices from container, preserving order of remaining elements, modifies container in-place
// assumes indices_to_remove is sorted in ascending order and contains valid indices
template <typename T>
void delete_by_index(std::vector<T>& container, const std::vector<int>& indices_to_remove)
{
    size_t num_items = container.size();

    if (indices_to_remove.size() != 0) {
        size_t write_idx = 0;
        size_t read_idx = 0;

        for (size_t delete_idx : indices_to_remove) {
            size_t count = delete_idx - read_idx;

            if (count > 0 && write_idx != read_idx) {
                std::copy(container.begin() + read_idx, container.begin() + delete_idx, container.begin() + write_idx);
            }

            write_idx += count;
            read_idx = delete_idx + 1;
        }

        if (read_idx < num_items && write_idx != read_idx) {
            std::copy(container.begin() + read_idx, container.end(), container.begin() + write_idx);
        }

        write_idx += num_items - read_idx;
        container.resize(write_idx);
    }
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

// Get input files from directory or wildcard pattern
static std::vector<std::filesystem::path> get_input_files(std::string& input) {
    std::vector<std::filesystem::path> filePaths;

    if (fs::is_directory(input)) {
        for (const auto& entry : fs::directory_iterator(input)) {
            if (entry.is_regular_file()) {
                filePaths.push_back(entry.path());
            }
        }
    }
    else {
        // check if has wildcard character (i.e., '*')
        if (input.find('*') != std::string::npos) {
            std::regex regx(replaceAll(fs::path(input).filename().string(), "*", ".*"));
            fs::path current_path(input);

            for (const auto& entry : fs::directory_iterator(current_path.parent_path())) {
                if (entry.is_regular_file() && std::regex_match(entry.path().filename().string(), regx)) {
                    filePaths.push_back(entry.path());
                }
            }
        }
        else {
            filePaths.push_back(fs::path(input));
        }
    }

    std::sort(filePaths.begin(), filePaths.end());
    return filePaths;
}

#define ENABLE_ZLIB_GZIP 32

static std::string read_file(const std::string& filename) {
    std::ifstream file_in(filename, std::ios_base::in);
    if (!file_in.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    std::ostringstream ss;
    ss << file_in.rdbuf();
    file_in.close();
    return ss.str();
}

// quick and dirty function to read gz file into string
static std::string read_gz_file(const std::string& filename) {
    std::string output;

    z_stream strm;
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;

	// read entire file into vector
    std::ifstream file_in(filename, std::ios_base::in |  std::ios_base::binary);
    if (!file_in.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    size_t file_size;
    std::vector<unsigned char> buffer;

    file_in.seekg(0, std::ios::end);
    file_size = file_in.tellg();
    file_in.seekg(0, std::ios::beg);
    buffer.resize(file_size);
    file_in.read(reinterpret_cast<char*>(buffer.data()), file_size);
    file_in.close();

    strm.avail_in = file_size;
    strm.next_in = buffer.data();

    if (inflateInit2(&strm, MAX_WBITS | ENABLE_ZLIB_GZIP) != Z_OK) {
        return output;
    }

    const int CHUNK = 1024;
    unsigned char out[CHUNK];

    // inflate
    do {
        strm.avail_out = CHUNK;
        strm.next_out = out;

        int ret = inflate(&strm, Z_NO_FLUSH);

        if (ret != Z_OK && ret != Z_STREAM_END) {
            inflateEnd(&strm);
            return output;
        }

        unsigned have = CHUNK - strm.avail_out;
        output.insert(output.end(), out, out + have);
    } while (strm.avail_out == 0);

    inflateEnd(&strm);
    return output;
}
