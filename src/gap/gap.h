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


template <typename RmpSolver>
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
    GapCompact lp;

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

    template <typename PricerType, typename FarkasPricerType> 
    int solve(PricerType& pricer, FarkasPricerType& pricer_farkas);

    template <typename PricerType, typename FarkasPricerType>
    int solve() {
        PricerType pricer;
        FarkasPricerType pricer_farkas;
		return solve(pricer, pricer_farkas);
    }

    template <typename FarkasPricerType>
    bool restoreFeasibility(FarkasPricerType &pricer_farkas);
	void updateCompactSolution();

    int add_columns(std::vector<double>& reduced_costs);
    double remove_duplicates();
};

template <typename RmpSolver, template<typename> class PricerType, template<typename> class FarkasPricerType>
int solve_gap_with_farkas(GapSolver<RmpSolver>& m) {
    return m.template solve<PricerType<RmpSolver>, FarkasPricerType<RmpSolver>>();
}

template <typename RmpSolver, template<typename> class FarkasPricerType>
int solve_gap_with_method(GapSolver<RmpSolver>& m, const std::string& pricing_method) {
    if (pricing_method == "mip_template") {
        return solve_gap_with_farkas<RmpSolver, TemplatePrice, FarkasPricerType>(m);
    }
    else if (pricing_method == "lagrange_template") {
        return solve_gap_with_farkas<RmpSolver, LagrangeTemplatePrice, FarkasPricerType>(m);
    }
    else if (pricing_method == "wentges") {
        return solve_gap_with_farkas<RmpSolver, WentgesPrice, FarkasPricerType>(m);
    }
    else if (pricing_method == "dantzig") {
        return solve_gap_with_farkas<RmpSolver, DantzigPrice, FarkasPricerType>(m);
    }
    else {
        std::cerr << "Unsupported pricing method: " << pricing_method << std::endl;
        return 0;
    }
}

template <typename RmpSolver>
int solve_gap(const std::string& filename, quill::CsvWriter<CsvSchema, quill::FrontendOptions>& csv_writer, std::string& pricing_method, std::string& init_method, Parameters& params) {
    GapSolver<RmpSolver> m(filename, params, csv_writer);

    // Determine the Farkas pricer type based on the init method
    if (init_method == "mip_template" || (init_method == "auto" && pricing_method == "mip_template")) {
        return solve_gap_with_method<RmpSolver, TemplateFarkas>(m, pricing_method);
    }
    else if (init_method == "lagrange_template" || (init_method == "auto" && pricing_method == "lagrange_template")) {
        return solve_gap_with_method<RmpSolver, LagrangeTemplateFarkas>(m, pricing_method);
    }
    else if (init_method == "dantzig") {
        return solve_gap_with_method<RmpSolver, DantzigFarkas>(m, pricing_method);
    }
    else {
        std::cerr << "Unsupported init method: " << init_method << std::endl;
        return 0;
    }
}