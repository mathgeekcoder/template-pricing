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

namespace fs = std::filesystem;

int gap_dantzig(const fs::path& filename, quill::CsvWriter<CsvSchema, quill::FrontendOptions>& csv_writer) {
    Parameters params;
    std::cout << std::endl << filename.filename().string() << " Dantzig" << std::endl;
	GapSolver(filename.string(), params, csv_writer).solve<DantzigPrice, DantzigFarkas>();
    return 0;
}

int gap_wentges_template(const fs::path& filename, quill::CsvWriter<CsvSchema, quill::FrontendOptions>& csv_writer) {
    Parameters params;
    std::cout << std::endl << filename.filename().string() << " WentgesTemplate" << std::endl;
    GapSolver(filename.string(), params, csv_writer).solve<WentgesTemplatePrice, WentgesTemplateFarkas>();
    return 0;
}

int gap_wentges(const fs::path& filename, quill::CsvWriter<CsvSchema, quill::FrontendOptions>& csv_writer) {
    Parameters params;
    std::cout << std::endl << filename.filename().string() << " Wentges" << std::endl;
    GapSolver(filename.string(), params, csv_writer).solve<WentgesPrice, DantzigFarkas>();
    return 0;
}

int gap_template(const fs::path& filename, quill::CsvWriter<CsvSchema, quill::FrontendOptions>& csv_writer) {
    Parameters params;
    TemplatePrice pricer;
    TemplateFarkas pricer_farkas;
    std::cout << std::endl << filename.filename().string() << " Template" << std::endl;
	GapSolver(filename.string(), params, csv_writer).solve<TemplatePrice, TemplateFarkas>(pricer, pricer_farkas);

 //   std::cout << std::endl << filename.filename().string() << " FixedTemplate" << std::endl;
 //   FixedTemplatePrice fixed_pricer;
 //   fixed_pricer._template = pricer._template;

 //   FixedTemplateFarkas fixed_farkas;
	//fixed_farkas._template = pricer._template;

 //   GapSolver(filename.string(), params, csv_writer).solve<FixedTemplatePrice, FixedTemplateFarkas>(fixed_pricer, fixed_farkas);
    return 0;
}

// An example of reoptimization within a branch-and-price framework.
int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cout << "no input provided" << std::endl;
        return 0;
    }

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

#ifdef _DEBUG
    highs::parallel::initialize_scheduler(1);
#else
	highs::parallel::initialize_scheduler(std::thread::hardware_concurrency());
#endif

    std::string pricing_method = "template";
    if (argc > 2)
        pricing_method = argv[2];

    quill::BackendOptions backend_options;
    quill::Backend::start(backend_options);

    fs::path current_path(argv[1]);

    for (const fs::path& filename : filePaths) {
        quill::CsvWriter<CsvSchema, quill::FrontendOptions> csv_writer(filename.filename().string() + "-" + pricing_method + "-output.csv");

        if (pricing_method == "template")
            gap_template(filename, csv_writer);
        else if (pricing_method == "dantzig")
            gap_dantzig(filename, csv_writer);
        else if (pricing_method == "wentges_template")
            gap_wentges_template(filename, csv_writer);
        else if (pricing_method == "wentges")
            gap_wentges(filename, csv_writer);
    }

    return 0;
}
