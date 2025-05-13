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

//#define RUN_PARALLEL
namespace fs = std::filesystem;

template <typename Pricer, typename PricerFarkas>
int run_gap(const Parameters& params, const fs::path& filename, quill::CsvWriter<CsvSchema, quill::FrontendOptions>& csv_writer) {
    std::cout << filename.filename().string() << " " + std::string(Pricer::name) << std::endl;
    return GapSolver(filename.string(), params, csv_writer).solve<Pricer, PricerFarkas>();
}


// An example of reoptimization within a branch-and-price framework.
int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cout << "no input provided" << std::endl;
        return 0;
    }

    HandleCtrlC::Enable();
    std::vector<fs::path> filePaths;

    if (fs::is_directory(argv[1])) {
        for (const auto& entry : fs::directory_iterator(argv[1])) {
            if (entry.is_regular_file()) {
                filePaths.push_back(entry.path());
            }
        }
    }
    else {
        // check if has wildcard character (i.e., '*')
        if (std::string(argv[1]).find('*') != std::string::npos) {
            std::string pattern = replaceAll(fs::path(argv[1]).filename().string(), "*", ".*");

            std::regex regx(pattern);
            fs::path current_path(argv[1]);

            for (const auto& entry : fs::directory_iterator(current_path.parent_path())) {
                if (entry.is_regular_file() && std::regex_match(entry.path().filename().string(), regx)) {
					filePaths.push_back(entry.path());
				}
			}
		}
        else {
            filePaths.push_back(fs::path(argv[1]));
        }
    }

    std::sort(filePaths.begin(), filePaths.end());

#ifndef NDEBUG
    highs::parallel::initialize_scheduler(1);
#else
	highs::parallel::initialize_scheduler(std::thread::hardware_concurrency());
#endif

    std::string pricing_method = "mip_template";
    if (argc > 2)
        pricing_method = argv[2];

	int REPLICATIONS = 1;
	if (argc > 3)
        REPLICATIONS = std::max(1, std::stoi(argv[3]));

    quill::BackendOptions backend_options;
    quill::Backend::start(backend_options);

    fs::path current_path(argv[1]);

#ifdef RUN_PARALLEL
    highs::parallel::for_each(0, filePaths.size(), [&](HighsInt start, HighsInt end) {
		for (int i = start; i < end; ++i) {
			const fs::path& filename = filePaths[i];
#else
        for (const fs::path& filename : filePaths) {
#endif
            int retries = 5;

            for (int replication = 1; replication <= REPLICATIONS; ++replication) {
                std::cout << std::endl << "Replication: " << replication << std::endl;
                quill::CsvWriter<CsvSchema, quill::FrontendOptions> csv_writer(filename.filename().string() + "-" + pricing_method + "-output-" + std::to_string(replication) + ".csv");

                int result = 0;
                Parameters params;
                params.random_seed = replication;// static_cast<int>(std::time(nullptr)) + replication;
				std::cout << "Random seed: " << params.random_seed << std::endl;

                if (pricing_method == "mip_template")
                    result = run_gap<TemplatePrice, TemplateFarkas>(params, filename, csv_writer);
                else if (pricing_method == "wentges_template")
                    result = run_gap<WentgesTemplatePrice, WentgesTemplateFarkas>(params, filename, csv_writer);
                else if (pricing_method == "wentges")
                    result = run_gap<WentgesPrice, DantzigFarkas>(params, filename, csv_writer);
                else if (pricing_method == "dantzig") {
                    params.min_col_factor = 3;
                    params.max_col_factor = 6;
                    result = run_gap<DantzigPrice, DantzigFarkas>(params, filename, csv_writer);
                }

                // if failure: repeat replication
                if (result != 0) {
                    if (--retries < 0) {
                        exit(-1);
                    }
                    --replication;
                }
            }
        }
#ifdef RUN_PARALLEL
    });
#endif

    quill::Backend::stop();
    return 0;
}
