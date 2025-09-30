#include <iostream>
#include <iomanip>
#include <algorithm>
#include <filesystem>
#include <sstream>
#include <fstream>
#include <regex>
#include <locale>
#include "parameters.h"
#include "gap/gap.h"
#include "quill/Backend.h"
#include <ctime>
#include "argparse/argparse.hpp"
#include "taskflow/taskflow.hpp"
#include "taskflow/algorithm/for_each.hpp"

namespace fs = std::filesystem;
bool has_completed(const std::string& log_filename);

template <typename RmpSolver>
int solve_gap(const fs::path& filename, quill::CsvWriter<CsvSchema, quill::FrontendOptions>& csv_writer, std::string& pricing_method, int seed, int replication, std::string keep_cols, int num_threads) {
    if (keep_cols == "best") {
        if (pricing_method == "lagrange_template") {
            keep_cols = "low";
        }
        else if (pricing_method == "mip_template" || pricing_method == "wentges") {
            keep_cols = "med";
        }
        else if (pricing_method == "dantzig") {
            keep_cols = "high";
        }
    }

    Parameters params;
    params.random_seed = (seed == -1 ? static_cast<int>(std::time(nullptr)) + replication : seed);
    params.num_threads = num_threads;
    params.column_retention = keep_cols;
	params.replication = replication;

    if (keep_cols == "low") {
        params.age_limit = 5;
        params.max_col_multiplier = 1.5;
    }
    else if (keep_cols == "med") {
        params.age_limit = 10;
        params.max_col_multiplier = 2;
    }
    else if (keep_cols == "high") {
        params.age_limit = 250;
        params.max_col_multiplier = 5;
    }
    else {
        std::cerr << "Unsupported column retention level: " << keep_cols << std::endl;
        return -1;
    }

	std::cout << "Solver: " << (std::is_same<RmpSolver, Highs>() ? "HiGHS" : "Gurobi") << std::endl;
    std::cout << "Replication: " << replication << std::endl;
    std::cout << "Random seed: " << params.random_seed << std::endl;
    std::cout << "Pricing method: " << pricing_method << std::endl;
    std::cout << "Column retention: " << keep_cols << std::endl;
    std::cout << filename.filename().string() << std::endl << std::endl;

    GapSolver<RmpSolver> m(filename.string(), params, csv_writer);

    if (pricing_method == "mip_template") {
        return m.solve<TemplatePrice<RmpSolver>, TemplateFarkas<RmpSolver>>();
    }
    else if (pricing_method == "lagrange_template") {
        return m.solve<LagrangeTemplatePrice<RmpSolver>, LagrangeTemplateFarkas<RmpSolver>>();
    }
    else if (pricing_method == "wentges") {
        return m.solve<WentgesPrice<RmpSolver>, DantzigFarkas<RmpSolver>>();
    }
    else if (pricing_method == "dantzig") {
        return m.solve<DantzigPrice<RmpSolver>, DantzigFarkas<RmpSolver>>();
    }
    else {
        std::cerr << "Unsupported pricing method: " << pricing_method << std::endl;
        return 0;
    }
}

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("colgen_pricing");

    program.add_argument("input")
        .help("Input file or directory (can use wildcard patterns)");

    program.add_argument("-f", "--force")
        .default_value(false)
        .implicit_value(true)
        .help("force overwrite of existing log files")
		.nargs(0);

    program.add_argument("-g", "--gurobi")
        .default_value(false)
        .implicit_value(true)
        .help("use Gurobi solver")
        .nargs(0);

    program.add_argument("-m", "--method")
        .default_value(std::string("lagrange_template"))
        .help("pricing method: {mip_template, lagrange_template, wentges, dantzig}")
        .choices("mip_template", "lagrange_template", "wentges", "dantzig")
        .nargs(1);

    program.add_argument("-k", "--keep_cols")
        .default_value(std::string("best"))
        .choices("low", "med", "high", "best")
        .help("column retention level: {low, med, high, best}")
        .nargs(1);

    program.add_argument("-s", "--seed")
        .default_value(-1)
        .scan<'i', int>()
        .help("random seed (-1 for system time)")
        .nargs(1);

    program.add_argument("-r", "--replications")
        .default_value(1)
        .scan<'i', int>()
        .help("number of replications")
        .nargs(1);

    program.add_argument("-p", "--parallel")
        .default_value(1)
        .scan<'i', int>()
        .help("number of parallel instances")
        .nargs(1);

    program.add_argument("-t", "--threads")
        .default_value(int(std::thread::hardware_concurrency()))
        .scan<'i', int>()
        .help("number of threads (per parallel instance)")
        .nargs(1);

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    std::string input = program.get<std::string>("input");
    std::string method = program.get<std::string>("--method");
    std::string keep_cols = program.get<std::string>("--keep_cols");

	bool force = program.get<bool>("--force");
	bool use_gurobi = program.get<bool>("--gurobi");
	int seed = program.get<int>("--seed");
    int num_replications = program.get<int>("--replications");
    int num_parallel = program.get<int>("--parallel");
    int num_threads = program.get<int>("--threads");

    if (num_parallel < 1) {
        std::cerr << "Number of parallel instances must be at least 1." << std::endl;
        return 1;
	}
    else if (num_threads < 1) {
        std::cerr << "Number of threads must be at least 1." << std::endl;
        return 1;
	}    
    else if (num_replications < 1) {
        std::cerr << "Number of replications must be at least 1." << std::endl;
        return 1;
	}

    HandleCtrlC::Enable();

	// Get input file paths
    std::vector<fs::path> filePaths;

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

    if (use_gurobi) {
#ifdef SUPPORT_GUROBI
        GurobiHighs::start();
#else
        std::cerr << "Gurobi support has not been enabled." << std::endl;
        return 1;
#endif
    }

#ifndef NDEBUG
    num_parallel = 1;
    num_threads = 1;
#endif

	tf::Executor executor(num_parallel);
	tf::Taskflow taskflow;

    quill::BackendOptions backend_options;
    quill::Backend::start(backend_options);

    fs::path current_path(input);

    for (int replication = 1; replication <= num_replications; ++replication) {
        taskflow.for_each(filePaths.begin(), filePaths.end(), [&, replication](const fs::path& filename) {
			std::string log_filename = std::format("{}-{}-{}-output-{}.csv", filename.filename().string(), method, keep_cols, replication);
            
            // check to see if output already exists, and if has been completed
            if (!force && has_completed(log_filename)) {
                std::cout << "Skipping " << filename.filename().string() << " (log exists and completed): " << log_filename << std::endl;
                return;
			}

            quill::CsvWriter<CsvSchema, quill::FrontendOptions> csv_writer(log_filename);

            int result = 0;

            // if failure: repeat replication (e.g., basis bug in HiGHS)
            for (int retries = 20; retries >= 0; --retries) {
                // GAP instances
                if (filename.extension() == "") {
                    if (use_gurobi) {
#ifdef SUPPORT_GUROBI
                        result = solve_gap<GurobiHighs>(filename, csv_writer, method, (seed == -1 ? (20 - retries) * 100 + (replication - 1) : seed), replication, keep_cols, num_threads);
#endif
                    }
                    else {
                        result = solve_gap<Highs>(filename, csv_writer, method, (seed == -1 ? (20 - retries) * 100 + (replication - 1) : seed), replication, keep_cols, num_threads);
					}

                    std::cout << std::endl;
                }
                else {
                    std::cerr << "Unsupported file format: " << filename.extension() << std::endl;
                    break;
                }

                if (result == 0) {
                    break;
                }
            }

            // delete output
            if (result != 0) {
                std::remove(log_filename.c_str());
            }
        });
    }

	executor.run(std::move(taskflow)).wait();

#ifdef SUPPORT_GUROBI
    if (use_gurobi) {
        GurobiHighs::stop();
    }
#endif

    quill::Backend::stop();
    return 0;
}

bool has_completed(const std::string& log_filename) {
    if (fs::exists(log_filename)) {
        try {
            std::ifstream in(log_filename);
            in.seekg(0, std::ios::end); // Move to the end of the file

            char ch;

            // Move backwards to find the last comma
            std::streamoff end_pos = static_cast<std::streamoff>(in.tellg());
            for (std::streamoff pos = end_pos - 1; pos >= 0; --pos) {
                in.seekg(static_cast<std::streampos>(pos));
                in.get(ch);
                if (ch == ',' && pos != end_pos - 1) {
                    break; // Found the last comma
                }
            }

            std::string last;
            std::getline(in, last);
            return (last == "1");
        }
        catch (...) {
            // If anything goes wrong reading the file, fall back to re-running
        }
    }
    return false;
}