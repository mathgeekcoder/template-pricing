#pragma once
#include <string>
#include <variant>
#include "parameters.h"
#include "block/column_generation.h"
#include "gap_compact.h"
#include "gap/gap_instance.h"
#include "quill/CsvWriter.h"
#include "quill/core/FrontendOptions.h"
#include "highs/Highs.h"
#include "pricers/dantzig.h"
#include "pricers/wentges.h"
#include "pricers/mip_template.h"
#include "pricers/lagrange_template.h"
#include "pricers/farkas.h"
#include "gap_lagrangian.h"
#include <format>

// phase I initializers
template <typename RmpSolver>
using FarkasVariant = std::variant<
    DantzigFarkas<RmpSolver>,
    TemplateFarkas<RmpSolver>,
    LagrangeTemplateFarkas<RmpSolver>
>;

// pricing algorithms
template <typename RmpSolver>
using PricerVariant = std::variant<
    DantzigPrice<RmpSolver>,
    WentgesPrice<RmpSolver>,
    TemplatePrice<RmpSolver>,
    LagrangeTemplatePrice<RmpSolver>
>;

// status information
enum class LogStatus {
    Iteration = 0,
    ValidTermination = 1,
    Gap = 2,
    Optimal = 3,
    Feasible = -1,
    TimeLimit = -4,
    UserInterrupt = -5,
    CompactOptimal = 6,
    FatalError = -7
};

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
    bool should_stop = false;

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

    PricerVariant<RmpSolver> _pricer;
    FarkasVariant<RmpSolver> _farkas;
    const char* _pricer_name;
    const char* _farkas_name;

	GapSolver(std::string filename, const Parameters& params, quill::CsvWriter<CsvSchema, quill::FrontendOptions>& csv_writer,
              FarkasVariant<RmpSolver> farkas, PricerVariant<RmpSolver> pricer)
		: instance(filename), params(params), csv_writer(csv_writer), pricing(instance.machines), lp(instance), _executor(params.num_threads), _farkas(std::move(farkas)), _pricer(std::move(pricer))
    {
        _farkas_name = std::visit([](const auto& f) { return f.name; }, _farkas);
        _pricer_name = std::visit([](const auto& p) { return p.name; }, _pricer);

		_ones.assign(instance.jobs + 1, 1);
		_reduced_costs.assign(instance.machines, 0);
		_compact_solution.assign(instance.jobs * instance.machines, 0);

		tbl.show_output = params.show_output;
		tbl.add_column("#Its", 7);
		tbl.add_column("LB", 14, 2);
		tbl.add_column("UB", 12, 0);
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

    template <typename FarkasType, typename PricerType>
    int solve_impl(FarkasType& farkas, PricerType& pricer);

    template <typename FarkasType>
    bool restoreFeasibility(FarkasType& farkas);
	void updateCompactSolution();

    int add_columns(std::vector<double>& reduced_costs);
    double remove_duplicates();

    void write_log(
        LogStatus status,
        const std::optional<double> gap = std::nullopt,
        const std::optional<double> rmp_obj = std::nullopt,
        const std::optional<double> total_reduced_cost = std::nullopt,
        const std::optional<int> basis_size = std::nullopt,
        const std::optional<double> pivots_per_sec = std::nullopt,
        const std::optional<double> pivots_per_col = std::nullopt)
    {
		bool include_params = status == LogStatus::ValidTermination || status == LogStatus::Feasible;

        csv_writer.append_row(
            instance.name,
            instance.name[0],
            instance.machines,
            instance.jobs,
            _pricer_name,
            params.solver,
            _farkas_name,
            params.replication,
            iteration_count,
            _LB,
            _UB,
            gap.has_value() ? std::to_string(*gap) : "",
            rmp_obj.has_value() ? std::to_string(*rmp_obj) : "",
            total_reduced_cost.has_value() ? std::to_string(*total_reduced_cost) : "",
            basis_size.value_or(0),
            rmp ? rmp->getNumCol() : 0,
            rmp_time.TotalSeconds(),
            cg_time.TotalSeconds(),
            total_time.TotalSeconds(),
            lp_iteration_count,
            pivots_per_sec.has_value() ? std::to_string(*pivots_per_sec) : "",
            pivots_per_col.has_value() ? std::to_string(*pivots_per_col) : "",
            status == LogStatus::Iteration || gap == std::nullopt ? int(basis_size == instance.machines) : int(!std::isnan(*gap)),
            (params.time_limit > 0 && total_time.TotalSeconds() > params.time_limit) ? 1 : 0,
            include_params ? params.to_json() : "",
            static_cast<int>(status)
        );
    }
};

template <typename RmpSolver>
PricerVariant<RmpSolver> make_pricer(const Parameters& params) {
    switch (params.pricing_method[0]) {
        case 'm': return TemplatePrice<RmpSolver>{};
        case 'l': return LagrangeTemplatePrice<RmpSolver>{};
        case 'w': return WentgesPrice<RmpSolver>{};
        default:  return DantzigPrice<RmpSolver>{};
    }
}

template <typename RmpSolver>
FarkasVariant<RmpSolver> make_farkas(const Parameters& params) {
    switch (params.init_method[0]) {
        case 'm': return TemplateFarkas<RmpSolver>{};
        case 'l': return LagrangeTemplateFarkas<RmpSolver>{};

        // "auto" — fall through to match pricing_method
        case 'a': 
            switch (params.pricing_method[0]) {
                case 'm': return TemplateFarkas<RmpSolver>{};
                case 'l': return LagrangeTemplateFarkas<RmpSolver>{};
                default:  return DantzigFarkas<RmpSolver>{};
            }
        default: return DantzigFarkas<RmpSolver>{};
    }
}

template <typename RmpSolver>
int solve_gap_solver(const std::string& filename, quill::CsvWriter<CsvSchema, quill::FrontendOptions>& csv_writer, Parameters& params) {
    // Determine the Farkas pricer type based on the init method
    if (params.pricing_method == "mip") {
        GapInstance instance(filename);
		instance.save_original(std::format("{}.orig", instance.name));
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
    else if (params.pricing_method == "lr") {
		// solve using lagrangian relaxation
		GapInstance instance(filename);
		GapLagrangian lagrangian_solver(instance, params);

        lagrangian_solver.solve(csv_writer);
        return 0;
    }
    else {
        GapSolver<RmpSolver> m(filename, params, csv_writer, make_farkas<RmpSolver>(params), make_pricer<RmpSolver>(params));
        return m.solve();
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
