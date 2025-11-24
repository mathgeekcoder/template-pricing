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

template <typename RmpSolver>
int solve_gap(const fs::path& filename, quill::CsvWriter<CsvSchema, quill::FrontendOptions>& csv_writer, std::string& pricing_method, Parameters& params) {
    //if (params.column_retention == "best") {
    //    if (pricing_method == "lagrange_template") {
    //        params.column_retention = "low";
    //    }
    //    else if (pricing_method == "mip_template" || pricing_method == "wentges") {
    //        params.column_retention = "med";
    //    }
    //    else if (pricing_method == "dantzig") {
    //        params.column_retention = "high";
    //    }
    //}

    //if (params.column_retention == "low") {
    //    params.age_limit = 5;
    //    params.max_col_multiplier = 1.5;
    //}
    //else if (params.column_retention == "med") {
    //    params.age_limit = 10;
    //    params.max_col_multiplier = 2;
    //}
    //else if (params.column_retention == "high") {
    //    params.age_limit = 250;
    //    params.max_col_multiplier = 5;
    //}
    //else if (params.column_retention != "custom") {
    //    std::cerr << "Unsupported column retention level: " << params.column_retention << std::endl;
    //    return -1;
    //}

	std::cout << "Solver: " << params.solver << std::endl;
    std::cout << "Replication: " << params.replication << std::endl;
    std::cout << "Random seed: " << params.random_seed << std::endl;
    std::cout << "Pricing method: " << pricing_method << std::endl;
    std::cout << "Age Limit: " << params.age_limit << std::endl;
    std::cout << "Column retention: " << params.column_retention << std::endl;
    std::cout << filename.filename().string() << std::endl << std::endl;

    GapSolver<RmpSolver> m(filename.string(), params, csv_writer);

    if (pricing_method == "mip_template") {
        return m.template solve<TemplatePrice<RmpSolver>, TemplateFarkas<RmpSolver>>();
    }
    else if (pricing_method == "lagrange_template") {
        return m.template solve<LagrangeTemplatePrice<RmpSolver>, LagrangeTemplateFarkas<RmpSolver>>();
    }
    else if (pricing_method == "wentges") {
        return m.template solve<WentgesPrice<RmpSolver>, DantzigFarkas<RmpSolver>>();
    }
    else if (pricing_method == "dantzig") {
        return m.template solve<DantzigPrice<RmpSolver>, DantzigFarkas<RmpSolver>>();
    }
    else {
        std::cerr << "Unsupported pricing method: " << pricing_method << std::endl;
        return 0;
    }
}

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("colgen_pricing");
    program.set_usage_max_line_width(80);
	program.add_description("Column Generation for the Generalized Assignment Problem with Various Pricing Methods");

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

    program.add_argument("--time_limit")
		.default_value(21600) // 6 hours
        .scan<'i', int>()
        .help("time limit (seconds) for each instance")
		.nargs(1);

	// parameter sweep options
	program.add_group("Parameter Sweep Options");

	 program.add_argument("--age_limit_start")
	 	.default_value(std::optional<int>())
	 	.scan<'i', int>()
	 	.help("starting age limit for column retention sweep")
	 	.nargs(1);
	 program.add_argument("--age_limit_end")
	 	.default_value(std::optional<int>())
	 	.scan<'i', int>()
	 	.help("ending age limit for column retention sweep")
	 	.nargs(1);
	 program.add_argument("--age_limit_step")
	 	.default_value(std::optional<int>())
	 	.scan<'i', int>()
	 	.help("step size for age limit in column retention sweep")
	 	.nargs(1);
	 program.add_argument("--multiplier_start")
	 	.default_value(std::optional<double>())
	 	.scan<'g', double>()
	 	.help("starting column multiplier for column retention sweep")
	 	.nargs(1);
	 program.add_argument("--multiplier_end")
	 	.default_value(std::optional<double>())
	 	.scan<'g', double>()
	 	.help("ending column multiplier for column retention sweep")
	 	.nargs(1);
	 program.add_argument("--multiplier_step")
	 	.default_value(std::optional<double>())
	 	.scan<'g', double>()
	 	.help("step size for column multiplier in column retention sweep")
	 	.nargs(1);

	 // take as a parameter a math function "--func", e.g., 0.0711*x^2 - 0.6111*x + 1
	 // where x is the jobs/machines ratio
	 program.add_argument("--func")
		 .default_value(std::optional<std::string>())
		 .help("function for determining column retention parameters based on jobs/machines ratio")
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

	// build lambda function for --func parameter
    std::optional<std::function<void(Parameters&, int, int)>> func = std::nullopt;
     if (program.is_used("--func")) {
         std::string func_str = program.get<std::string>("--func");
         func = [func_str](Parameters& params, int machines, int jobs) {
             // very simple parser for quadratic functions of the form ax^2 + bx + c
             // where x is jobs/machines ratio
             double a = 0.0, b = 0.0, c = 0.0;
             std::regex regex(R"(([+-]?\d*\.?\d*)\*?x\^2\s*([+-]\s*\d*\.?\d*)\*?x\s*([+-]\s*\d*\.?\d*))");
             std::smatch match;
             if (std::regex_search(func_str, match, regex)) {
                 a = std::stod(replaceAll(match[1].str(), " ", ""));
                 b = std::stod(replaceAll(match[2].str(), " ", ""));
                 c = std::stod(replaceAll(match[3].str(), " ", ""));
             }
             double ratio = static_cast<double>(jobs) / static_cast<double>(machines);
             double result = a * ratio * ratio + b * ratio + c;
             params.max_col_multiplier = 1;
             params.age_limit = std::max(1, static_cast<int>(std::ceil(result - 1e-6)));
         };
	 }

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

	// if parameter sweep options provided
    if (!parameter_sweep(program, filePaths, taskflow)) {
        for (int replication = 1; replication <= num_replications; ++replication) {
            taskflow.for_each(filePaths.begin(), filePaths.end(), [&, replication](const fs::path& filename) {
                std::string log_filename = std::format("{}-{}-{}-output-{}-{}.csv", filename.filename().string(), method, keep_cols, replication, use_gurobi ? "gurobi" : "highs");

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
                        Parameters params;
                        params.time_limit = program.get<int>("--time_limit");
                        params.random_seed = (seed == -1 ? (20 - retries) * 100 + (replication - 1) : seed);
                        params.num_threads = num_threads;
                        params.column_retention = keep_cols;
                        params.replication = replication;
                        strcpy(params.solver, (use_gurobi ? "gurobi" : "highs"));

                        if (func.has_value()) {
                            // load instance to get machines/jobs
                            GapInstance gap_instance(filename.string());
                            (*func)(params, gap_instance.machines, gap_instance.jobs);
                        }

                        if (use_gurobi) {
#ifdef SUPPORT_GUROBI
                            result = solve_gap<GurobiHighs>(filename, csv_writer, method, params);
#endif
                        }
                        else {
                            result = solve_gap<Highs>(filename, csv_writer, method, params);
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
    bool any = (program.is_used("--age_limit_start") || program.is_used("--age_limit_end") || program.is_used("--age_limit_step") ||
        program.is_used("--multiplier_start") || program.is_used("--multiplier_end") || program.is_used("--multiplier_step"));

    bool all = (program.is_used("--age_limit_start") && program.is_used("--age_limit_end") && program.is_used("--age_limit_step") &&
        program.is_used("--multiplier_start") && program.is_used("--multiplier_end") && program.is_used("--multiplier_step"));

    if (any) {
        if (!all) {
            std::cerr << "Must provide all parameter sweep options." << std::endl;
            exit(-1);
		}
        
        int age_limit_start = program.get<int>("--age_limit_start");
        int age_limit_end = program.get<int>("--age_limit_end");
        int age_limit_step = program.get<int>("--age_limit_step");
        double multiplier_start = program.get<double>("--multiplier_start");
        double multiplier_end = program.get<double>("--multiplier_end");
        double multiplier_step = program.get<double>("--multiplier_step");

        if (age_limit_start < 1 || age_limit_end < age_limit_start || age_limit_step < 1) {
            std::cerr << "Invalid age limit sweep parameters." << std::endl;
            exit(-1);
        }

        // parameter sweep for column retention
        for (int replication = 1; replication <= program.get<int>("--replications"); ++replication) {
            for (int age_limit = age_limit_start; age_limit <= age_limit_end; age_limit += age_limit_step) {
                for (double max_col_multiplier = multiplier_start; max_col_multiplier <= multiplier_end; max_col_multiplier += multiplier_step) {

                    taskflow.for_each(filePaths.begin(), filePaths.end(), [&, replication, age_limit, max_col_multiplier](const fs::path& filename) {
                        Parameters params;
                        params.random_seed = (program.get<int>("--seed") == -1 ? (replication - 1) : program.get<int>("--seed"));
                        params.time_limit = program.get<int>("--time_limit");
                        params.num_threads = program.get<int>("--threads");
                        params.replication = replication;
                        strcpy(params.solver, (program.get<bool>("--gurobi") ? "gurobi" : "highs"));
                        std::string method = program.get<std::string>("--method");

                        params.column_retention = "custom";
                        params.age_limit = age_limit;
                        params.max_col_multiplier = max_col_multiplier;
                        std::cout << "Age limit: " << params.age_limit << ", Column multiplier: " << params.max_col_multiplier << std::endl;

                        std::string log_filename = std::format("{}-{}-{}-output-{}-{}-{}-{}.csv",
                            filename.filename().string(), method, params.column_retention, params.replication, params.solver, params.age_limit, params.max_col_multiplier);

                        // check to see if output already exists, and if has been completed
                        if (!program.get<bool>("--force") && has_completed(log_filename)) {
                            std::cout << "Skipping " << filename.filename().string() << " (log exists and completed): " << log_filename << std::endl;
                            return;
                        }

                        quill::CsvWriter<CsvSchema, quill::FrontendOptions> csv_writer(log_filename);

                        if (program.get<bool>("--gurobi")) {
#ifdef SUPPORT_GUROBI
                            solve_gap<GurobiHighs>(filename, csv_writer, method, params);
#endif
                        }
                        else {
                            solve_gap<Highs>(filename, csv_writer, method, params);
                        }

                        std::cout << std::endl;
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