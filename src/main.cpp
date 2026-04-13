#include <iostream>
#include <algorithm>
#include <filesystem>
#include <sstream>
#include <fstream>

#include "parameters.h"
#include "gap/gap.h"
#include "quill/Backend.h"
#include "argparse/argparse.hpp"
#include "taskflow/taskflow.hpp"
#include "taskflow/algorithm/for_each.hpp"

namespace fs = std::filesystem;
bool has_completed(const std::string& log_filename);
bool parameter_sweep(argparse::ArgumentParser& program, std::vector<fs::path>& filePaths, tf::Taskflow& taskflow);
std::function<void(Parameters&, int, int)> column_retention_func(const std::vector<double>& params);

std::function<void(Parameters&, int, int)> quadratic_column_retention_func(double a, double b, double c) {
    return [a, b, c](Parameters& params, int machines, int jobs) {
        double ratio = static_cast<double>(jobs) / static_cast<double>(machines);
        params.max_col_multiplier = 1;
        params.age_limit = std::max(1, static_cast<int>(std::ceil((a * ratio + b) * ratio + c - 1e-6)));
    };
}

static const std::unordered_map<std::string, std::vector<double>> default_retention_params = {
    {"p",  {0.0, 0.3, 1.0}},
    {"d",  {0.081875, 0.0, 1.0}},
    {"lt", {0.00044, 0.0405, 1.0}},
    {"mt", {0.00044, 0.0405, 1.0}},
    {"mip", {0, 0, 0}},
	{"lr", {0, 0, 0}}
};

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

    program.add_argument("--farkas_only")
        .flag()
        .help("only perform RMP initialization");

    program.add_argument("-g", "--gurobi")
        .flag()
        .help("use Gurobi solver");

    program.add_argument("--set_partition")
        .flag()
        .help("use set partition RMP (instead of cover)");

    program.add_argument("-m", "--method")
        .default_value(std::string("lt"))
        .metavar("METHOD")
        .help("pricer method: {mt, lt, p, d, mip, lr}")
        .choices("mt", "lt", "p", "d", "mip", "lr");

    program.add_argument("-i", "--init")
        .default_value(std::string("auto"))
        .metavar("METHOD")
        .help("farkas method: {mt, lt, d, auto}")
        .choices("mt", "lt", "d", "auto");

    program.add_argument("-s", "--seed")
        .default_value(-1)
        .metavar("N")
        .scan<'i', int>()
        .help("random seed (-1 for system time)");

    program.add_argument("-r", "--replications")
        .default_value(1)
        .metavar("N")
        .scan<'i', int>()
        .help("number of replications");

    program.add_argument("-p", "--parallel")
        .default_value(1)
        .metavar("N")
        .scan<'i', int>()
        .help("number of parallel instances");

    program.add_argument("-t", "--threads")
        .default_value(int(std::thread::hardware_concurrency()))
        .metavar("N")
        .scan<'i', int>()
        .help("number of threads (per parallel instance)");

    program.add_argument("--time_limit")
        .default_value(21600)
        .metavar("SECS")
        .scan<'i', int>()
        .help("time limit (seconds) for each instance");

	// parameter sweep options
	program.add_group("Retention Options");

	 program.add_argument("--age_sweep")
	 	.nargs(3)
	 	.metavar("START END STEP")
	 	.scan<'i', int>()
	 	.help("age limit sweep parameters");

	 program.add_argument("--multiplier_sweep")
         .nargs(3)
         .metavar("START END STEP")
         .scan<'g', double>()
         .help("column multiplier sweep parameters");

     // take as a parameter a math function "--func", e.g., 0.0711*x^2 - 0.6111*x + 1
     // where x is the jobs/machines ratio
     program.add_argument("--func")
        .nargs(3)
        .metavar("A B C")
        .scan<'g', double>()
        .help("quadratic function coefficients a, b, c for ax^2+bx+c (e.g., 0.07 -0.6 1)");

     try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl << std::endl;
        std::cerr << program;
        return 1;
    }

    std::string input = program.get<std::string>("input");
    std::string pricing_method = program.get<std::string>("--method");
    std::string init_method = program.get<std::string>("--init");

	bool force = program.get<bool>("--force");
	bool farkas_only = program.get<bool>("--farkas_only");
	bool use_gurobi = program.get<bool>("--gurobi");
	bool set_partition = program.get<bool>("--set_partition");

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
    std::vector<double> retention_params = program.is_used("--func") ? program.get<std::vector<double>>("--func") : default_retention_params.at(pricing_method);
    auto set_age_limit = column_retention_func(retention_params);

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
                std::string log_filename = std::format("{}-{}-{}-output-{}-{}.csv", filename.filename().string(), pricing_method, init_method, replication, use_gurobi ? "gurobi" : "highs");

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
							params.nodes = (farkas_only ? 0 : 1);
							params.set_partition = set_partition;
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
    bool any = (program.is_used("--age_sweep") || program.is_used("--multiplier_sweep"));
    bool all = (program.is_used("--age_sweep") && program.is_used("--multiplier_sweep"));

    if (any) {
        if (!all) {
            std::cerr << "Must provide all parameter sweep options." << std::endl;
            exit(-1);
		}
        
        auto age_params = program.get<std::vector<int>>("--age_sweep");
        int age_start = age_params[0];
        int age_end = age_params[1];
        int age_step = age_params[2];

        auto multiplier_params = program.get<std::vector<double>>("--multiplier_sweep");
        double multiplier_start = multiplier_params[0];
        double multiplier_end = multiplier_params[1];
        double multiplier_step = multiplier_params[2];

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


std::function<void(Parameters&, int, int)> column_retention_func(const std::vector<double>& params) {
    if (params.size() == 3) {
        return quadratic_column_retention_func(params[0], params[1], params[2]);
    }
    else {
        throw std::invalid_argument("Invalid number of parameters for column retention function.");
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