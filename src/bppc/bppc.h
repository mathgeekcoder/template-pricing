#pragma once
#include <string>
#include "parameters.h"
#include "block/column_generation.h"
#include "bppc_compact.h"
#include "bppc/bppc_instance.h"
#include "bppc/bppc_pricing.h"
#include "quill/CsvWriter.h"
#include "quill/core/FrontendOptions.h"
#include "highs/parallel/HighsParallel.h"
#include "highs/Highs.h"
#include "bppc/pricers/dantzig.h"
#include "bppc/pricers/mip_template.h"

struct BppcSolver {
    static constexpr int ITERATION_OUTPUT = 1;
    static constexpr double ITERATION_TIME = 0.1;

    int basis_size = 0;
    bool isIntegral = false;
    size_t iteration_count = 0;
    size_t lp_iteration_count = 0;
    size_t fractional_count = 0;
    size_t prune_count = 0;
    size_t leaf_count = 0;
    double _LB = -kHighsInf;
    double _UB = kHighsInf;
    double previous_logging_time = -1;
    Timer total_time, rmp_time, cg_time;

    std::unique_ptr<Highs> rmp;

    const Parameters& params;
    BppcInstance instance;

    quill::CsvWriter<CsvSchema, quill::FrontendOptions>& csv_writer;
    std::vector<double> _compact_solution;
    std::vector<double> _compact_solution_best;
    std::vector<double> _ones;
    std::vector<double> _reduced_costs;

    ColumnAlignOutput tbl;
    PricingBlockVector<BppcPricing> pricing;

    BppcSolver(std::string filename, const Parameters& params, quill::CsvWriter<CsvSchema, quill::FrontendOptions>& csv_writer)
        : instance(filename), params(params), csv_writer(csv_writer), pricing(100) {

        _ones.assign(instance.items + 1, 1);
        //_compact_solution.assign(instance.jobs * instance.machines, 0);
        //_compact_solution_best.assign(instance.jobs * instance.machines, 0);

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

    template <typename PricerType> 
    int solve(PricerType& pricer);

    template <typename PricerType>
    int solve() {
        PricerType pricer;
        return solve(pricer);
    }

    void updateCompactSolution();
    bool add_columns(std::vector<double>& reduced_costs);
};