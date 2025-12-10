#pragma once
#include <string>
#include "parameters.h"
#include "block/column_generation.h"
#include "gap_compact.h"
#include "gap/gap_instance.h"
#include "quill/CsvWriter.h"
#include "quill/core/FrontendOptions.h"
#include "extern/scip/scip_knapsack.h"
#include "highs/parallel/HighsParallel.h"
#include "highs/Highs.h"
#include "pricers/dantzig.h"
#include "pricers/wentges.h"
#include "pricers/mip_template.h"
#include "pricers/lagrange_template.h"
#include "pricers/farkas.h"
#include "taskflow/taskflow.hpp"

template <typename RmpSolver>
struct AgeColumnManagement {
    RmpSolver* _rmp = nullptr;
    const GapInstance* _instance = nullptr;
    const Parameters* _params = nullptr;
    std::vector<uint32_t> _age;

    void init(RmpSolver* rmp, const GapInstance* instance, const Parameters& params) {
        _rmp = rmp;
        _instance = instance;
        _age.resize(_rmp->getNumCol(), 0);
        _params = &params;
    }

    void reduce(uint32_t iteration_count);
};


template <typename RmpSolver, 
          template<typename> class FarkasType, 
          template<typename> class PricerType>
struct GapSolver {
    static constexpr int ITERATION_OUTPUT = 5;
    static constexpr double ITERATION_TIME = 1;

    uint32_t basis_size = 0;
    uint32_t iteration_count = 0;
    size_t lp_iteration_count = 0;
    size_t fractional_count = 0;
    double _LB = -kHighsInf;
    double _UB = kHighsInf;
    double previous_logging_time = -1;
    Timer total_time, rmp_time, cg_time;

    tf::Executor _executor;
    std::unique_ptr<RmpSolver> rmp;

    const Parameters& params;
    GapInstance instance;

    quill::CsvWriter<CsvSchema, quill::FrontendOptions>& csv_writer;
    std::vector<double> _compact_solution;
    std::vector<double> _ones;
    std::vector<double> _reduced_costs;

    ColumnAlignOutput tbl;
    AgeColumnManagement<RmpSolver> column_management;
    PricingBlockVector<GapPricing> pricing;
    GapCompact<RmpSolver> lp;

    GapSolver(std::string filename, const Parameters& params, quill::CsvWriter<CsvSchema, quill::FrontendOptions>& csv_writer)
        : instance(filename), params(params), csv_writer(csv_writer), pricing(instance.machines), lp(instance), _executor(params.num_threads) {

        _ones.assign(instance.jobs + 1, 1);
        _reduced_costs.assign(instance.machines, 0);
        _compact_solution.assign(instance.jobs * instance.machines, 0);

        tbl.show_output = params.show_output;
        tbl.add_column("#Its", 7);
        tbl.add_column("LB", 14, 2);
        tbl.add_column("UB", 14, 2);
        tbl.add_column("Gap%", 9, 2);
        tbl.add_column("OBJ", 14, 2);
        tbl.add_column("Dfeas", 14, 2);
        tbl.add_column("#Basis", 10, 0);
        tbl.add_column("#Cols", 10, 0);
        tbl.add_column("Time", 10, 1);
        tbl.add_column("Lp #Its", 11);
        tbl.add_column("#Frac", 7);
    }

    void presolve();
    int solve();

    bool restoreFeasibility(FarkasType<RmpSolver> &pricer_farkas);
	void updateCompactSolution();

    int add_columns(std::vector<double>& reduced_costs);
    double remove_duplicates();
};

template <typename RmpSolver, template<typename> class PricerType, template<typename> class FarkasType>
int solve_gap_impl(const std::string& filename, quill::CsvWriter<CsvSchema, quill::FrontendOptions>& csv_writer, Parameters& params) {
    GapSolver<RmpSolver, FarkasType, PricerType> m(filename, params, csv_writer);
    return m.solve();
}

template <typename RmpSolver, template<typename> class FarkasPricerType>
int solve_gap_farkas(const std::string& filename, quill::CsvWriter<CsvSchema, quill::FrontendOptions>& csv_writer, Parameters& params) {
    if (params.pricing_method == "mip_template") {
        return solve_gap_impl<RmpSolver, TemplatePrice, FarkasPricerType>(filename, csv_writer, params);
    }
    else if (params.pricing_method == "lagrange_template") {
        return solve_gap_impl<RmpSolver, LagrangeTemplatePrice, FarkasPricerType>(filename, csv_writer, params);
    }
    else if (params.pricing_method == "wentges") {
        return solve_gap_impl<RmpSolver, WentgesPrice, FarkasPricerType>(filename, csv_writer, params);
    }
    else if (params.pricing_method == "dantzig") {
        return solve_gap_impl<RmpSolver, DantzigPrice, FarkasPricerType>(filename, csv_writer, params);
    }
    else {
        std::cerr << "Unsupported pricing method: " << params.pricing_method << std::endl;
        return 0;
    }
}

template <typename RmpSolver>
int solve_gap_solver(const std::string& filename, quill::CsvWriter<CsvSchema, quill::FrontendOptions>& csv_writer, Parameters& params) {
    // Determine the Farkas pricer type based on the init method
    if (params.pricing_method == "mip") {
        GapInstance instance(filename);
        GapCompact<RmpSolver> compact(instance);
        int stop_count = 5;

        HandleCtrlC ctrl_c_handler([&]() {
            if (stop_count <= 0) {
                exit(-1);
            }

            std::cout << std::format("Ctrl-C pressed, stopping... Press {} times to force stop\n", stop_count);
            compact.terminate();
            --stop_count;
        });

        compact.solve(true, true);
        return 0;
    }
    else if (params.init_method == "mip_template" || (params.init_method == "auto" && params.pricing_method == "mip_template")) {
        return solve_gap_farkas<RmpSolver, TemplateFarkas>(filename, csv_writer, params);
    }
    else if (params.init_method == "lagrange_template" || (params.init_method == "auto" && params.pricing_method == "lagrange_template")) {
        return solve_gap_farkas<RmpSolver, LagrangeTemplateFarkas>(filename, csv_writer, params);
    }
    else if (params.init_method == "dantzig" || params.init_method == "auto") {
        return solve_gap_farkas<RmpSolver, DantzigFarkas>(filename, csv_writer, params);
    }
    else {
        std::cerr << "Unsupported init method: " << params.init_method << std::endl;
        return 0;
    }
}

static int solve_gap(const std::string& filename, quill::CsvWriter<CsvSchema, quill::FrontendOptions>& csv_writer, Parameters& params) {
    if (params.solver[0] == 'g') {
#ifdef SUPPORT_GUROBI
        return solve_gap_solver<GurobiHighs>(filename, csv_writer, params);
#else
        std::cerr << "Gurobi not supported." << std::endl;
        return 0;
#endif
    }
    else {
        return solve_gap_solver<Highs>(filename, csv_writer, params);
    }
}
