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
#include <optional>

namespace fs = std::filesystem;
bool has_completed(const std::string& log_filename);
bool parameter_sweep(argparse::ArgumentParser& program, std::vector<fs::path>& filePaths, tf::Taskflow& taskflow);
std::function<void(Parameters&, int, int)> column_retention_func(const std::string method);

std::function<void(Parameters&, int, int)> quadratic_column_retention_func(double a, double b, double c) {
    return [a, b, c](Parameters& params, int machines, int jobs) {
        double ratio = static_cast<double>(jobs) / static_cast<double>(machines);
        params.max_col_multiplier = 1;
        params.age_limit = std::max(1, static_cast<int>(std::ceil((a * ratio + b) * ratio + c - 1e-6)));
    };
}

int solve_gap_instance(const fs::path& filename, std::string& log_filename, Parameters& params) {
    int result = 0;
    quill::CsvWriter<CsvSchema, quill::FrontendOptions> csv_writer(log_filename);

    std::cout << "Solver: " << params.solver << std::endl
              << "#Rep  : " << params.replication << std::endl
              << "Seed  : " << params.random_seed << std::endl
              << "Age   : " << params.age_limit << std::endl
              << "Farkas: " << params.init_method << std::endl
              << "Pricer: " << params.pricing_method << std::endl
              << "Inst  : " << filename.filename().string() << std::endl << std::endl;

    result = solve_gap(filename.string(), csv_writer, params);

    std::cout << std::endl;
	return result;
}

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("colgen_pricing");
    program.set_usage_max_line_width(80);
	program.add_description("Column Generation for the Generalized Assignment Problem with Various Pricing Methods");

    program.add_argument("input")
        .help("Input file or directory (can use wildcard patterns)");

    program.add_argument("-f", "--force")
        .flag()
        .help("force overwrite of existing log files");

    program.add_argument("-g", "--gurobi")
        .flag()
        .help("use Gurobi solver");

    program.add_argument("-m", "--method")
        .default_value(std::string("lagrange_template"))
        .help("pricing method: {mip_template, lagrange_template, wentges, dantzig, mip}")
        .choices("mip_template", "lagrange_template", "wentges", "dantzig", "mip");

    program.add_argument("-i", "--init")
        .default_value(std::string("auto"))
        .help("initialization pricing method: {mip_template, lagrange_template, dantzig, auto}")
        .choices("mip_template", "lagrange_template", "dantzig", "auto");

    program.add_argument("-s", "--seed")
        .default_value(-1)
        .scan<'i', int>()
        .help("random seed (-1 for system time)");

    program.add_argument("-r", "--replications")
        .default_value(1)
        .scan<'i', int>()
        .help("number of replications");

    program.add_argument("-p", "--parallel")
        .default_value(1)
        .scan<'i', int>()
        .help("number of parallel instances");

    program.add_argument("-t", "--threads")
        .default_value(int(std::thread::hardware_concurrency()))
        .scan<'i', int>()
        .help("number of threads (per parallel instance)");

    program.add_argument("--time_limit")
		.default_value(21600) // 6 hours
        .scan<'i', int>()
        .help("time limit (seconds) for each instance");

    // take as a parameter a math function "--func", e.g., 0.0711*x^2 - 0.6111*x + 1
    // where x is the jobs/machines ratio
    program.add_argument("--func")
        .default_value(std::optional<std::string>())
        .help("function for determining column retention parameters based on jobs/machines ratio");

	// parameter sweep options
	program.add_group("Parameter Sweep Options");

	 program.add_argument("--age_start")
	 	.scan<'i', int>()
	 	.help("starting age limit for column retention sweep");

	 program.add_argument("--age_end")
	 	.scan<'i', int>()
	 	.help("ending age limit for column retention sweep");

     program.add_argument("--age_step")
         .scan<'i', int>()
         .help("step size for age limit in column retention sweep");

	 program.add_argument("--multiplier_start")
	 	.scan<'g', double>()
	 	.help("starting column multiplier for column retention sweep");

	 program.add_argument("--multiplier_end")
	 	.scan<'g', double>()
	 	.help("ending column multiplier for column retention sweep");

	 program.add_argument("--multiplier_step")
	 	.scan<'g', double>()
	 	.help("step size for column multiplier in column retention sweep");

     try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    std::string input = program.get<std::string>("input");
    std::string pricing_method = program.get<std::string>("--method");
    std::string init_method = program.get<std::string>("--init");

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

    // build lambda function for --func parameter (or "method" defaults)
    auto set_age_limit = column_retention_func(program.is_used("--func") ? program.get<std::string>("--func") : pricing_method);

	// Get input file paths
    std::vector<fs::path> filePaths = get_input_files(input);

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

    HandleCtrlC::Enable();
    tf::Executor executor(num_parallel);
	tf::Taskflow taskflow;

    quill::BackendOptions backend_options;
    quill::Backend::start(backend_options);

    fs::path current_path(input);

	// if parameter sweep options provided
    if (!parameter_sweep(program, filePaths, taskflow)) {
        for (int replication = 1; replication <= num_replications; ++replication) {
            taskflow.for_each(filePaths.begin(), filePaths.end(), [&, replication](const fs::path& filename) {
                std::string log_filename = std::format("{}-{}-best-output-{}-{}.csv", filename.filename().string(), pricing_method, replication, use_gurobi ? "gurobi" : "highs");

                // check to see if output already exists, and if has been completed
                if (!force && has_completed(log_filename)) {
                    std::cout << "Skipping " << filename.filename().string() << " (log exists and completed): " << log_filename << std::endl;
                    return;
                }

                // GAP instances
				std::vector<std::string> supported_extensions = { ".gz", ".json", "" };

                try {
					bool supported = std::any_of(supported_extensions.begin(), supported_extensions.end(),
                        [&](const std::string& ext) {
                            if (filename.extension() == ext) {
                                return true;
                            }
                            return false;
						});

                    if (supported) {
                        int result = 0;

                        // if failure: repeat replication (e.g., basis bug in HiGHS)
                        for (int retries = 20; retries >= 0; --retries) {
                            Parameters params;
                            params.time_limit = program.get<int>("--time_limit");
                            params.random_seed = (seed == -1 ? (20 - retries) * 100 + (replication - 1) : seed);
                            params.num_threads = num_threads;
                            params.replication = replication;
                            strcpy(params.solver, (use_gurobi ? "gurobi" : "highs"));
                            params.init_method = init_method;
                            params.pricing_method = pricing_method;

                            // load instance to get machines/jobs
                            GapInstance gap_instance(filename.string());
                            set_age_limit(params, gap_instance.machines, gap_instance.jobs);

						    result = solve_gap_instance(filename, log_filename, params);

                            if (result == 0) {
                                break;
                            }
                        }

                        // delete output
                        if (result != 0) {
                            std::remove(log_filename.c_str());
                        }
                    }
                    else {
                        std::cerr << "Unsupported file format: " << filename.extension() << std::endl;
                    }
                }
                catch (const std::exception& e) {
                    std::cerr << "Error processing file " << filename.filename().string() << ": " << e.what() << std::endl;
				}
            });
        }
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

bool parameter_sweep(argparse::ArgumentParser& program, std::vector<fs::path>& filePaths, tf::Taskflow& taskflow)
{
    bool any = (program.is_used("--age_start") || program.is_used("--age_end") || program.is_used("--age_step") ||
        program.is_used("--multiplier_start") || program.is_used("--multiplier_end") || program.is_used("--multiplier_step"));

    bool all = (program.is_used("--age_start") && program.is_used("--age_end") && program.is_used("--age_step") &&
        program.is_used("--multiplier_start") && program.is_used("--multiplier_end") && program.is_used("--multiplier_step"));

    if (any) {
        if (!all) {
            std::cerr << "Must provide all parameter sweep options." << std::endl;
            exit(-1);
		}
        
        int age_start = program.get<int>("--age_start");
        int age_end = program.get<int>("--age_end");
        int age_step = program.get<int>("--age_step");

        double multiplier_start = program.get<double>("--multiplier_start");
        double multiplier_end = program.get<double>("--multiplier_end");
        double multiplier_step = program.get<double>("--multiplier_step");

        if (age_start < 1 || age_end < age_start || age_step < 1) {
            std::cerr << "Invalid age limit sweep parameters." << std::endl;
            exit(-1);
        }

        // parameter sweep for column retention
        for (int replication = 1; replication <= program.get<int>("--replications"); ++replication) {
            for (int age_limit = age_start; age_limit <= age_end; age_limit += age_step) {
                for (double max_col_multiplier = multiplier_start; max_col_multiplier <= multiplier_end; max_col_multiplier += multiplier_step) {

                    taskflow.for_each(filePaths.begin(), filePaths.end(), [&, replication, age_limit, max_col_multiplier](const fs::path& filename) {
                        Parameters params;
                        params.random_seed = (program.get<int>("--seed") == -1 ? (replication - 1) : program.get<int>("--seed"));
                        params.time_limit = program.get<int>("--time_limit");
                        params.num_threads = program.get<int>("--threads");
                        params.replication = replication;
                        strcpy(params.solver, (program.get<bool>("--gurobi") ? "gurobi" : "highs"));
                        params.init_method = program.get<std::string>("--init");
                        params.pricing_method = program.get<std::string>("--method");

                        params.age_limit = age_limit;
                        params.max_col_multiplier = max_col_multiplier;
                        std::cout << "Age limit: " << params.age_limit << ", Column multiplier: " << params.max_col_multiplier << std::endl;

                        std::string log_filename = std::format("{}-{}-custom-output-{}-{}-{}-{}.csv",
                            filename.filename().string(), params.pricing_method, params.replication, params.solver, params.age_limit, params.max_col_multiplier);

                        // check to see if output already exists, and if has been completed
                        if (!program.get<bool>("--force") && has_completed(log_filename)) {
                            std::cout << "Skipping " << filename.filename().string() << " (log exists and completed): " << log_filename << std::endl;
                            return;
                        }

                        solve_gap_instance(filename, log_filename, params);
                    });
                }
            }
        }

        return true;
    }
    else
    {
        return false;
    }
}

std::function<void(Parameters&, int, int)> column_retention_func(const std::string method) {
    if (method == "wentges") {
        return quadratic_column_retention_func(0, 0.3, 1);
    }
    else if (method == "dantzig") {
        return quadratic_column_retention_func(0.081875, 0, 1);
    }
    else if (method == "lagrange_template" || method == "mip_template") {
        return quadratic_column_retention_func(0.00044, 0.0405, 1);
    }
    else {
        // very simple parser for quadratic functions of the form ax^2 + bx + c
        double a = 0.0, b = 0.0, c = 0.0;
        std::regex regex(R"(([+-]?\d*\.?\d*)\*?x\^2\s*([+-]\s*\d*\.?\d*)\*?x\s*([+-]\s*\d*\.?\d*))");
        std::smatch match;
        if (std::regex_search(method, match, regex)) {
            a = std::stod(replaceAll(match[1].str(), " ", ""));
            b = std::stod(replaceAll(match[2].str(), " ", ""));
            c = std::stod(replaceAll(match[3].str(), " ", ""));
        }

        return quadratic_column_retention_func(a, b, c);
    }
}

// check if log file indicates completed run (ends with ",1")
// this will not be the case if the program crashed or was interrupted (Ctrl-C)
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