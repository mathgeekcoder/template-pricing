#include <vector>
#include <algorithm>
#include "gap.h"
#include "block/column_generation.h"
#include "highs/util/HighsIntegers.h"
#include <numeric>
#include <format>

#include "gap_instance.h"
#include <filesystem>
#include "utils.h"

// provide additional status information
enum class LogStatus {
    Iteration = 0,
	ValidTermination = 1,
    Gap = 2,
    Optimal = 3,
    Feasible = -1,
    TimeLimit = -4,
    UserInterrupt = -5,
    CompactOptimal = 6
};

// Helper to apply RmpSolver template parameter to a list of pricer templates
template <typename RmpSolver, template<typename> class... Pricers>
using RmpTuple = std::tuple<Pricers<RmpSolver>...>;

// Pricer types
template <typename RmpSolver>
using PricerTypes = RmpTuple<RmpSolver, 
    DantzigPrice, 
    WentgesPrice, 
    TemplatePrice, 
    LagrangeTemplatePrice
>;

// Farkas types
template <typename RmpSolver>
using FarkasPricerTypes = RmpTuple<RmpSolver, 
    DantzigFarkas, 
    TemplateFarkas, 
    LagrangeTemplateFarkas
>;


template class GapSolver<Highs, DantzigFarkas, DantzigPrice>;
template class GapSolver<Highs, DantzigFarkas, WentgesPrice>;
template class GapSolver<Highs, DantzigFarkas, TemplatePrice>;
template class GapSolver<Highs, DantzigFarkas, LagrangeTemplatePrice>;

template class GapSolver<Highs, TemplateFarkas, DantzigPrice>;
template class GapSolver<Highs, TemplateFarkas, WentgesPrice>;
template class GapSolver<Highs, TemplateFarkas, TemplatePrice>;
template class GapSolver<Highs, TemplateFarkas, LagrangeTemplatePrice>;

template class GapSolver<Highs, LagrangeTemplateFarkas, DantzigPrice>;
template class GapSolver<Highs, LagrangeTemplateFarkas, WentgesPrice>;
template class GapSolver<Highs, LagrangeTemplateFarkas, TemplatePrice>;
template class GapSolver<Highs, LagrangeTemplateFarkas, LagrangeTemplatePrice>;

#ifdef SUPPORT_GUROBI

template class GapSolver<GurobiHighs, DantzigFarkas, DantzigPrice>;
template class GapSolver<GurobiHighs, DantzigFarkas, WentgesPrice>;
template class GapSolver<GurobiHighs, DantzigFarkas, TemplatePrice>;
template class GapSolver<GurobiHighs, DantzigFarkas, LagrangeTemplatePrice>;

template class GapSolver<GurobiHighs, TemplateFarkas, DantzigPrice>;
template class GapSolver<GurobiHighs, TemplateFarkas, WentgesPrice>;
template class GapSolver<GurobiHighs, TemplateFarkas, TemplatePrice>;
template class GapSolver<GurobiHighs, TemplateFarkas, LagrangeTemplatePrice>;

template class GapSolver<GurobiHighs, LagrangeTemplateFarkas, DantzigPrice>;
template class GapSolver<GurobiHighs, LagrangeTemplateFarkas, WentgesPrice>;
template class GapSolver<GurobiHighs, LagrangeTemplateFarkas, TemplatePrice>;
template class GapSolver<GurobiHighs, LagrangeTemplateFarkas, LagrangeTemplatePrice>;

#endif


template <typename RmpSolver>
void AgeColumnManagement<RmpSolver>::reduce(uint32_t iteration_count) {
    // "age" the columns, i.e., set the current basis to the current iteration
    const auto& solution = _rmp->getSolution();
    const HighsInt* basis = _rmp->getBasicVariablesArray();
    const uint32_t num_col = _rmp->getNumCol();
    uint32_t basis_size = 0;
    _age.resize(num_col, iteration_count - 1);

    for (uint32_t i = 0; i < _rmp->getNumRow(); ++i) {
        uint32_t idx = basis[i];

        // include all basis columns, even if they have zero value
        // this helps reduce number of simplex pivots
        if (idx < num_col) {
            bool non_zero = solution.col_value[idx] > 1e-6;
            _age[idx] = iteration_count + non_zero;
            basis_size += non_zero;
        }
    }

    // remove old columns if we've got too many, but only if we have made progress
    const uint32_t MAX_COLS = _params->max_col_multiplier * _rmp->getNumRow();

    if (num_col > MAX_COLS && iteration_count > _params->age_limit) {
		const uint32_t age_limit = iteration_count - _params->age_limit;
        std::vector<int> indices_to_remove;

		for (uint32_t i = 0; i < num_col; ++i) {
            if (_age[i] < age_limit) {
                indices_to_remove.emplace_back(i);
            }
		}

        // remove the columns from model and from _age, reverse order to preserve correct index
		for (auto it = indices_to_remove.crbegin(); it != indices_to_remove.crend(); ++it) {
			_age.erase(_age.begin() + *it);
		}
        _rmp->deleteCols(static_cast<int>(indices_to_remove.size()), indices_to_remove.data());
        //printf("  Reducing columns from %d -> %d [%d] (basis %d), iteration %d\n", num_col, num_col-indices_to_remove.size(), indices_to_remove.size(), basis_size, iteration_count);
    }
}

template <typename RmpSolver, template<typename> class FarkasType, template<typename> class PricerType>
void GapSolver<RmpSolver, FarkasType, PricerType>::presolve() {
    // solve LP for template pricing
    _LB = lp.solve();
    lp_iteration_count = lp.iterations;

    auto compact_solution = lp._solution;
    bool all_integer = true;

    for (size_t idx = 0, size = compact_solution.col_value.size(); idx < size && all_integer; ++idx) {
        all_integer &= HighsIntegers::isIntegral(compact_solution.col_value[idx], 1e-6);
    }

    // compact LP is integral optimal, so break
    if (all_integer == true) {
        _UB = _LB;
    }
}

template <typename RmpSolver, template<typename> class FarkasType, template<typename> class PricerType>
int GapSolver<RmpSolver, FarkasType, PricerType>::solve() {
    PricerType<RmpSolver> pricer;
    FarkasType<RmpSolver> pricer_farkas;

    std::string params_json = params.to_json();
    highs::parallel::initialize_scheduler(1);
    total_time.start();
    tbl.write_header();

    presolve();

	if (_UB == _LB) {
        std::cout << std::format("Compact solution is integer optimal.");

        // final entry
        csv_writer.append_row(instance.name, instance.name[0], instance.machines, instance.jobs, pricer.name, params.solver, pricer_farkas.name, params.replication, iteration_count,
            _LB, _UB, 0, _LB, 0, 0, 0, rmp_time.TotalSeconds(), cg_time.TotalSeconds(), total_time.TotalSeconds(),
            lp_iteration_count, "", "", 1, 0, params_json, (int)LogStatus::CompactOptimal);

        // final entry
        csv_writer.append_row(instance.name, instance.name[0], instance.machines, instance.jobs, pricer.name, params.solver, pricer_farkas.name, params.replication, iteration_count,
            _LB, _UB, 0, _LB, 0, 0, 0, rmp_time.TotalSeconds(), cg_time.TotalSeconds(), total_time.TotalSeconds(),
            lp_iteration_count, "", "", 1, 0, params_json, (int)LogStatus::ValidTermination);

        return 0;
	}

    HandleCtrlC ctrl_c_handler([&]() { 
		if (should_stop) exit(-1);  // force stop

		std::cout << std::format("{} {}: Ctrl-C pressed, stopping...\n", instance.name, pricer.name);
        should_stop = true; 
    });

    rmp.reset(new RmpSolver);
    rmp->setOptionValue("output_flag", false);
    rmp->setOptionValue(kPresolveString, "off");
    rmp->setOptionValue("random_seed", params.random_seed);
	rmp->setOptionValue("threads", 1);
	rmp->setOptionValue("simplex_strategy", "4"); // primal simplex
    std::function<HighsInt()> get_lp_iters = [&]() { return rmp->getInfo().simplex_iteration_count; };

    auto model = SetCoverRestrictedProblem(instance.jobs, instance.machines, ObjSense::kMinimize);
    //auto model = SetPartitionRestrictedProblem(instance.jobs, instance.machines, ObjSense::kMinimize);
    rmp->passModel(model);
    pricer.init(&_executor, rmp.get(), &instance);

    // initialize pricing
    pricing.init(instance);
    pricer_farkas.init(&_executor, pricer, pricing, lp);

    double optimal_pricing = 0.0;
	int added_columns = 1; // initialize to 1 to avoid NaN in first iteration
    column_management.init(rmp.get(), &instance, params);

    if (!restoreFeasibility(pricer_farkas)) {
		// assume user interrupt
        csv_writer.append_row(instance.name, instance.name[0], instance.machines, instance.jobs, pricer.name, params.solver, pricer_farkas.name, params.replication, iteration_count,
            _LB, _UB, "", "", "", "", rmp->getNumCol(), rmp_time.TotalSeconds(), cg_time.TotalSeconds(), total_time.TotalSeconds(),
            lp_iteration_count, "", "", 0, int((params.time_limit > 0 && total_time.TotalSeconds() > params.time_limit)), params_json, (int)LogStatus::UserInterrupt);

        std::cout << std::format("\n"
            "Inst : {}\n"
            "Price: {}\n"
            "RMP  : {:.3f} s\n"
            "CG   : {:.3f} s\n"
            "Total: {:.3f} s\n"
            "#Cols: {}\n"
            "Pvt/s: {:.2f}\n",
            instance.name, pricer.name, rmp_time.TotalSeconds(), cg_time.TotalSeconds(), total_time.TotalSeconds(), rmp->getNumCol(), lp_iteration_count / rmp_time.TotalSeconds());

        return 0;
    }

    updateCompactSolution();
    double _rmpLB = rmp->getObjectiveValue();

    tbl.output(iteration_count, _LB, _UB, "-", _rmpLB, "-", basis_size, rmp->getNumCol(), total_time.TotalSeconds(), lp_iteration_count, fractional_count);
    csv_writer.append_row(instance.name, instance.name[0], instance.machines, instance.jobs, pricer.name, params.solver, pricer_farkas.name, params.replication, iteration_count,
        _LB, _UB, "", _rmpLB, "", basis_size, rmp->getNumCol(), rmp_time.TotalSeconds(), cg_time.TotalSeconds(), total_time.TotalSeconds(), 
        lp_iteration_count, lp_iteration_count / rmp_time.TotalSeconds(), lp_iteration_count / static_cast<double>(rmp->getNumCol()), int(basis_size == instance.machines), int(false), params_json, (int)LogStatus::Feasible);

    // primal simplex for warm-start "add columns"
    //rmp->setOptionValue("simplex_strategy", "4");
    //rmp->setOptionValue("allow_unbounded_or_infeasible", false);  // not sure if this adds unnecessary overheads
    //rmp->setOptionValue(kSolverString, kIpmString);
    //rmp->setOptionValue(kRunCrossoverString, kHighsOffString);
    pricer.init_feasible();
    //column_management.init(rmp.get(), &instance, params);

    double avg_pivots_per_column = lp_iteration_count / static_cast<double>(rmp->getNumCol()) / static_cast<double>(iteration_count);

    if (params.nodes > 0) {
        do {
rmp_time.start();
            auto status = rmp->run();
rmp_time.pause();

            if (rmp->getModelStatus() != HighsModelStatus::kOptimal) {
                std::cout << std::format("{} {}: Error - {}\n", instance.name, pricer.name, (int)rmp->getModelStatus());
                return -1;
            }

            _rmpLB = rmp->getObjectiveValue();
            auto& solution = rmp->getSolution();

            updateCompactSolution();
            lp_iteration_count += get_lp_iters();
			double lp_iteration_per_column = get_lp_iters() / static_cast<double>(added_columns);

cg_time.start();
            pricer.update();
            optimal_pricing = pricer.optimize(solution.row_dual, pricing, _reduced_costs);
cg_time.pause();

            column_management.reduce(iteration_count);
            added_columns = add_columns(_reduced_costs);

            // ASSUMES lambda <= 1, otherwise need to scale (e.g. _rmpLB / (1 + reduced cost))
            // This is only valid in root node, otherwise need to keep track of worst node
            _LB = std::max(_LB, _rmpLB - optimal_pricing);

           // check if we can stop
            double lb = std::ceil(_LB - 1e-6);
            double gap = (_UB - lb) / _UB;

            if (gap < params.gap || lb + 1e-6 >= _rmpLB)
                break;

            // logging
            csv_writer.append_row(instance.name, instance.name[0], instance.machines, instance.jobs, pricer.name, params.solver, pricer_farkas.name, params.replication, iteration_count,
                _LB, _UB, gap * 100, _rmpLB, optimal_pricing, basis_size, rmp->getNumCol(), rmp_time.TotalSeconds(), cg_time.TotalSeconds(), total_time.TotalSeconds(), 
                lp_iteration_count, lp_iteration_count / rmp_time.TotalSeconds(), lp_iteration_per_column, int(basis_size==instance.machines), int(false), "", (int)LogStatus::Iteration);

            if (iteration_count % ITERATION_OUTPUT == 0 && total_time.TotalSeconds() - previous_logging_time > ITERATION_TIME) {
                tbl.output(iteration_count, _LB, _UB, gap * 100, _rmpLB, optimal_pricing, basis_size, rmp->getNumCol(), total_time.TotalSeconds(), lp_iteration_count, fractional_count);
                previous_logging_time = total_time.TotalSeconds();
            }

            ++iteration_count;
            avg_pivots_per_column = ((iteration_count - 1) * avg_pivots_per_column + lp_iteration_per_column) / static_cast<double>(iteration_count);

        } while (added_columns > 0 && (params.time_limit < 0 || total_time.TotalSeconds() < params.time_limit) && !should_stop);
    }

	double lb = std::ceil(_LB - 1e-6);
    double gap = (_UB - lb) / _UB;
    tbl.output(iteration_count, lb, _UB, gap * 100, _rmpLB, optimal_pricing, basis_size, rmp->getNumCol(), total_time.TotalSeconds(), lp_iteration_count, fractional_count);

	// provide additional status information
	LogStatus last_status = LogStatus::Iteration;

    if (gap < params.gap || lb + 1e-6 >= _rmpLB)
        last_status = LogStatus::Gap;
    else if (added_columns == 0)
        last_status = LogStatus::Optimal;
    else if (params.time_limit > 0 && total_time.TotalSeconds() > params.time_limit)
        last_status = LogStatus::TimeLimit;
    else if (should_stop)
		last_status = LogStatus::UserInterrupt;

    csv_writer.append_row(instance.name, instance.name[0], instance.machines, instance.jobs, pricer.name, params.solver, pricer_farkas.name, params.replication, iteration_count,
        _LB, _UB, std::abs(gap*100), _rmpLB, optimal_pricing, basis_size, rmp->getNumCol(), rmp_time.TotalSeconds(), cg_time.TotalSeconds(), total_time.TotalSeconds(),
        lp_iteration_count, lp_iteration_count / rmp_time.TotalSeconds(), avg_pivots_per_column, int(!std::isnan(gap)), int((params.time_limit > 0 && total_time.TotalSeconds() > params.time_limit)), params_json, (int)last_status);

	// final entry only if not user interrupted
    if (!should_stop) {
        csv_writer.append_row(instance.name, instance.name[0], instance.machines, instance.jobs, pricer.name, params.solver, pricer_farkas.name, params.replication, iteration_count,
            _LB, _UB, std::abs(gap * 100), _rmpLB, optimal_pricing, basis_size, rmp->getNumCol(), rmp_time.TotalSeconds(), cg_time.TotalSeconds(), total_time.TotalSeconds(),
            lp_iteration_count, lp_iteration_count / rmp_time.TotalSeconds(), avg_pivots_per_column, int(!std::isnan(gap)), int((params.time_limit > 0 && total_time.TotalSeconds() > params.time_limit)), params_json, (int)LogStatus::ValidTermination);
    }

    std::cout << std::format("\n"
		"Inst : {}\n"
		"Price: {}\n"
        "RMP  : {:.3f} s\n" 
        "CG   : {:.3f} s\n" 
		"Total: {:.3f} s\n" 
		"#Cols: {}\n"  
        "Gap  : {:.2f}% \n"
        "Pvt/s: {:.2f}\n", 
        instance.name, pricer.name, rmp_time.TotalSeconds(), cg_time.TotalSeconds(), total_time.TotalSeconds(), rmp->getNumCol(), gap, lp_iteration_count / rmp_time.TotalSeconds());

    return 0;
}

//
//template <typename RmpSolver, template<typename> class FarkasType, template<typename> class PricerType>
//bool GapSolver<RmpSolver, FarkasType, PricerType>::restoreFeasibility(FarkasType<RmpSolver>& pricer_farkas) {
//    bool has_dual_ray = false;
//    std::vector<double> dual_ray(rmp->getNumRow(), 1);
//
//    if constexpr (std::is_same<RmpSolver, Highs>::value) {
//        rmp->setOptionValue("simplex_strategy", "1"); // need to use dual solver for extreme ray
//    }
//    else {
//        rmp->setOptionValue("simplex_strategy", "4"); // use primal simplex for other solvers
//    }
//
//    rmp->setOptionValue("allow_unbounded_or_infeasible", true);
//
//    // initialize RMP if empty
//    cg_time.start();
//    if (rmp->getNumCol() == 0 && pricer_farkas.optimize(dual_ray, pricing, _reduced_costs) > -kHighsInf) {
//        add_columns(_reduced_costs);
//        ++iteration_count;
//    }
//    cg_time.pause();
//
//    // tight loop to restore feasibility (assuming possible!)
//    do {
//rmp_time.start();
//        auto status = rmp->run();
//        auto modelStatus = rmp->getModelStatus();
//
//        // HiGHs has gotten into a bad state, so we need to reset?
//        if (status == HighsStatus::kError) {
//            std::cout << std::format("{}: Unrecoverable Error {} {}\n", instance.name, (int)status, (int)modelStatus);
//            return false;
//        }
//
//        rmp->getDualRay(has_dual_ray, dual_ray.data());
//        lp_iteration_count += rmp->getInfo().simplex_iteration_count;
//
//        if (modelStatus == HighsModelStatus::kUnknown && has_dual_ray == false) {
//            // there's an issue with dual simplex when tabooing pivots to prove infeasibility
//            // it returns unknown status with no dual ray
//            std::cout << std::format("{}: Error proving infeasibility and calculating dual ray - {}\n", instance.name, (int)modelStatus);
//            return false;
//        }
//        rmp_time.pause();
//
//        cg_time.start();
//        if (has_dual_ray) {
//            if (pricer_farkas.optimize(dual_ray, pricing, _reduced_costs) > -kHighsInf) {
//                //column_management.reduce(iteration_count); 
//                add_columns(_reduced_costs);
//            }
//            else {
//                std::cout << "Node infeasible!" << std::endl;
//                return false;
//            }
//        }
//        cg_time.pause();
//
//        // debugging
//        if (iteration_count % ITERATION_OUTPUT == 0 && total_time.TotalSeconds() - previous_logging_time > ITERATION_TIME && has_dual_ray) {
//            tbl.output(iteration_count, _LB, "-", "-", "-", "-", "-", rmp->getNumCol(), total_time.TotalSeconds(), lp_iteration_count, 0);
//            previous_logging_time = total_time.TotalSeconds();
//        }
//
//        ++iteration_count;
//    } while (has_dual_ray == true);
//
//    return true;
//}

//template <typename RmpSolver, template<typename> class FarkasType, template<typename> class PricerType>
//bool GapSolver<RmpSolver, FarkasType, PricerType>::restoreFeasibility(FarkasType<RmpSolver>& pricer_farkas) {
//    bool has_dual_ray = false;
//    int offset = instance.jobs + 2 * instance.machines;
//
//    // phase I primal simplex to get dual ray
//    // modify with dummy variables, one per >= row, two per == row
//    // for set cover, # jobs >=, # machines == (so jobs+2xmachines)
//    std::vector<double> ones(offset, 1.0);
//    std::vector<double> zeros(offset, 0.0);
//    std::vector<double> upper(offset, kHighsInf);
//
//    std::vector<double> costs;
//
//    // the first #jobs rows have one dummy variable each, the next #machines have two each (+ve and -ve)
//    std::vector<int> start(offset + 1);
//    std::vector<int> index(offset);
//    std::vector<double> value(offset, 1.0);
//
//    std::iota(start.begin(), start.end(), 0);
//    std::iota(index.begin(), index.begin() + instance.jobs + instance.machines, 0);
//
//    for (int row = 0; row < instance.machines; ++row) {
//        value[instance.jobs + instance.machines + row] = -1.0;
//        index[instance.jobs + instance.machines + row] = instance.jobs + row;
//    }
//
//    auto status1 = rmp->addCols(offset, ones.data(), zeros.data(), upper.data(), index.size(), start.data(), index.data(), value.data());
//    rmp->run();
//
//    // initialize RMP if empty
//    cg_time.start();
//    if (pricer_farkas.optimize(rmp->getSolution().row_dual, pricing, _reduced_costs) > -kHighsInf) {
//        //add_columns(_reduced_costs);
//
//        for (int m = 0; m < instance.machines; ++m) {
//            if (_reduced_costs[m] > 1e-6) {
//                auto& solution = pricing[m].solution;
//                std::stable_sort(solution.begin(), solution.end()); // ensure columns are sorted for faster search
//                rmp->addCol(0, 0, kHighsInf, static_cast<int>(pricing[m].solution.size()), pricing[m].solution.data(), _ones.data());
//
//                costs.push_back(sum(solution.begin(), --solution.end(), 0.0, instance.profit[m]));
//            }
//        }
//
//        ++iteration_count;
//    }
//    cg_time.pause();
//
//    // tight loop to restore feasibility (assuming possible!)
//    do {
//rmp_time.start();
//        auto status = rmp->run();
//        HighsModelStatus modelStatus = HighsModelStatus::kOptimal;
//
//        double count_infeasibilities = rmp->getObjectiveValue();
//
//        if (rmp->getObjectiveValue() > 1e-6) {
//            modelStatus = HighsModelStatus::kInfeasible;
//        }
//
//        // HiGHs has gotten into a bad state, so we need to reset?
//        if (status == HighsStatus::kError) {
//            std::cout << std::format("{}: Unrecoverable Error {} {}\n", instance.name, (int)status, (int)modelStatus);
//            return false;
//        }
//
//        has_dual_ray = modelStatus == HighsModelStatus::kInfeasible;
//        lp_iteration_count += rmp->getInfo().simplex_iteration_count;
//
//        if (modelStatus == HighsModelStatus::kUnknown && has_dual_ray == false) {
//            // there's an issue with dual simplex when tabooing pivots to prove infeasibility
//            // it returns unknown status with no dual ray
//            std::cout << std::format("{}: Error proving infeasibility and calculating dual ray - {}\n", instance.name, (int)modelStatus);
//            return false;
//        }
//        rmp_time.pause();
//
//        // age the columns
//        const auto& solution = rmp->getSolution();
//        const HighsInt* basis = rmp->getBasicVariablesArray();
//        const uint32_t num_col = rmp->getNumCol();
//        uint32_t basis_size = 0;
//        column_management._age.resize(num_col - offset, iteration_count - 1);
//
//        for (uint32_t i = 0; i < rmp->getNumRow(); ++i) {
//            uint32_t idx = basis[i];
//
//            // include all basis columns, even if they have zero value
//            // this helps reduce number of simplex pivots
//            if (idx < num_col && idx > offset) {
//                bool non_zero = solution.col_value[idx] > 1e-6;
//                column_management._age[idx - offset] = iteration_count + non_zero;
//                basis_size += non_zero;
//            }
//        }
//
//        cg_time.start();
//        if (has_dual_ray) {
//            pricer_farkas.update(rmp.get());
//
//            if (pricer_farkas.optimize(rmp->getSolution().row_dual, pricing, _reduced_costs) > -kHighsInf) {
//                //column_management.reduce(iteration_count);
//                //add_columns(_reduced_costs);
//
//                for (int m = 0; m < instance.machines; ++m) {
//                    if (_reduced_costs[m] > 1e-6) {
//                        auto& solution = pricing[m].solution;
//                        std::stable_sort(solution.begin(), solution.end()); // ensure columns are sorted for faster search
//                        rmp->addCol(0, 0, kHighsInf, static_cast<int>(pricing[m].solution.size()), pricing[m].solution.data(), _ones.data());
//
//                        costs.push_back(sum(solution.begin(), --solution.end(), 0.0, instance.profit[m]));
//                    }
//                }
//            }
//            else {
//                std::cout << "Node infeasible!" << std::endl;
//                return false;
//            }
//        }
//        cg_time.pause();
//
//        // debugging
//        if (iteration_count % ITERATION_OUTPUT == 0 && total_time.TotalSeconds() - previous_logging_time > ITERATION_TIME && has_dual_ray) {
//            tbl.output(iteration_count, _LB, "-", "-", "-", "-", "-", rmp->getNumCol() - offset, total_time.TotalSeconds(), lp_iteration_count, -count_infeasibilities);
//            previous_logging_time = total_time.TotalSeconds();
//        }
//
//        ++iteration_count;
//    } while (has_dual_ray == true && should_stop == false);
//
//    if (!should_stop) {
//        // remove dummy variables
//        std::vector<int> indices_to_remove(offset);
//        std::iota(indices_to_remove.begin(), indices_to_remove.end(), 0);
//        rmp->deleteCols(static_cast<int>(indices_to_remove.size()), indices_to_remove.data());
//        rmp->changeColsCost(0, costs.size() - 1, costs.data());
//
//        rmp->run();
//        return true;
//    }
//    else {
//        return false;
//    }
//}



template <typename RmpSolver, template<typename> class FarkasType, template<typename> class PricerType>
bool GapSolver<RmpSolver, FarkasType, PricerType>::restoreFeasibility(FarkasType<RmpSolver>& pricer_farkas) {
    bool has_dual_ray = false;
    int offset = instance.jobs;

    // phase I primal simplex to get dual ray
	// modify with dummy variables, one per job (machines are covered by initial columns)
    std::vector<double> ones(instance.jobs + instance.machines, 1.0);
    std::vector<double> zeros(offset, 0.0);
    std::vector<double> upper(offset, kHighsInf);
    std::vector<double> costs;

    std::vector<int> start(offset + 1);
    std::vector<int> index(offset);

    std::iota(start.begin(), start.end(), 0);
    std::iota(index.begin(), index.end(), 0);

    rmp->addCols(offset, ones.data(), zeros.data(), upper.data(), index.size(), start.data(), index.data(), ones.data());
    rmp->run();

    // initialize RMP if empty
    cg_time.start();
    if (pricer_farkas.optimize(ones, pricing, _reduced_costs) > -kHighsInf) {
        int from = rmp->getNumCol();
        int added = add_columns(_reduced_costs);

		// capture costs of added columns and replace with zero cost for phase I
        costs.resize(costs.size() + added);
		getColsCost(*rmp, from, from + added - 1, &costs.back() - added + 1);
		rmp->changeColsCost(from, from + added - 1, zeros.data());
        ++iteration_count;
    }
    cg_time.pause();

    // tight loop to restore feasibility (assuming possible!)
    do {
        rmp_time.start();
        auto status = rmp->run();
        HighsModelStatus modelStatus = HighsModelStatus::kOptimal;

        double count_infeasibilities = rmp->getObjectiveValue();

        if (rmp->getObjectiveValue() > 1e-6) {
            modelStatus = HighsModelStatus::kInfeasible;
        }

        // HiGHs has gotten into a bad state, so we need to reset?
        if (status == HighsStatus::kError) {
            std::cout << std::format("{}: Unrecoverable Error {} {}\n", instance.name, (int)status, (int)modelStatus);
            return false;
        }

        has_dual_ray = modelStatus == HighsModelStatus::kInfeasible;
        lp_iteration_count += rmp->getInfo().simplex_iteration_count;

        if (modelStatus == HighsModelStatus::kUnknown && has_dual_ray == false) {
            // there's an issue with dual simplex when tabooing pivots to prove infeasibility
            // it returns unknown status with no dual ray
            std::cout << std::format("{}: Error proving infeasibility and calculating dual ray - {}\n", instance.name, (int)modelStatus);
            return false;
        }
        rmp_time.pause();

        // age the columns
        const auto& solution = rmp->getSolution();
        const HighsInt* basis = rmp->getBasicVariablesArray();
        const uint32_t num_col = rmp->getNumCol();
        uint32_t basis_size = 0;
        column_management._age.resize(num_col - offset, iteration_count - 1);

        for (uint32_t i = 0; i < rmp->getNumRow(); ++i) {
            uint32_t idx = basis[i];

            // include all basis columns, even if they have zero value
            // this helps reduce number of simplex pivots
            if (idx < num_col && idx > offset) {
                bool non_zero = solution.col_value[idx] > 1e-6;
                column_management._age[idx - offset] = iteration_count + non_zero;
                basis_size += non_zero;
            }
        }

        cg_time.start();
        if (has_dual_ray) {
            pricer_farkas.update(rmp.get());

            if (pricer_farkas.optimize(rmp->getSolution().row_dual, pricing, _reduced_costs) > -kHighsInf) {
                //column_management.reduce(iteration_count);
                int from = rmp->getNumCol();
                int added = add_columns(_reduced_costs);

                // capture costs of added columns and replace with zero cost for phase I
                costs.resize(costs.size() + added);
                getColsCost(*rmp, from, from + added - 1, &costs.back() - added + 1);
                rmp->changeColsCost(from, from + added - 1, zeros.data());
            }
            else {
                std::cout << "Node infeasible!" << std::endl;
                return false;
            }
        }
        cg_time.pause();

        // debugging
        if (iteration_count % ITERATION_OUTPUT == 0 && total_time.TotalSeconds() - previous_logging_time > ITERATION_TIME && has_dual_ray) {
            tbl.output(iteration_count, _LB, "-", "-", "-", "-", "-", rmp->getNumCol() - offset, total_time.TotalSeconds(), lp_iteration_count, -count_infeasibilities);
            previous_logging_time = total_time.TotalSeconds();
        }

        ++iteration_count;
    } while (has_dual_ray == true && should_stop == false);

    if (!should_stop) {
		// remove dummy variables and restore costs
        rmp->deleteCols(static_cast<int>(index.size()), index.data());
        rmp->changeColsCost(0, costs.size() - 1, costs.data());

        rmp->run();
        return true;
    }
    else {
        return false;
    }
}


//template <typename RmpSolver, template<typename> class FarkasType, template<typename> class PricerType>
//bool GapSolver<RmpSolver, FarkasType, PricerType>::restoreFeasibility(FarkasType<RmpSolver>& pricer_farkas) {
//    bool has_dual_ray = false;
//    int offset = instance.jobs + 2 * instance.machines;
//
//	// test a silly idea: reconstruct the RMP model with a phase I primal simplex to get dual ray
//    std::unique_ptr<RmpSolver> rmp_phase1;
//    rmp_phase1.reset(new RmpSolver);
//
//	rmp_phase1->setOptionValue("output_flag", false);
//	rmp_phase1->setOptionValue(kPresolveString, "off");
//	rmp_phase1->setOptionValue("random_seed", params.random_seed);
//	rmp_phase1->setOptionValue("threads", 1);
//    rmp_phase1->setOptionValue("simplex_strategy", "4");
//
//    auto model = SetCoverRestrictedProblem(instance.jobs, instance.machines, ObjSense::kMinimize);
//    rmp_phase1->passModel(model);
//	
//    // modify with dummy variables, one per >= row, two per == row
//	// for set cover, # jobs >=, # machines == (so jobs+2xmachines)
//	std::vector<double> ones(offset, 1.0);
//    std::vector<double> zeros(offset, 0.0);
//	std::vector<double> upper(offset, kHighsInf);
//	
//	// the first #jobs rows have one dummy variable each, the next #machines have two each (+ve and -ve)
//    std::vector<int> start(offset + 1);
//	std::vector<int> index(offset);
//	std::vector<double> value(offset, 1.0);
//
//	std::iota(start.begin(), start.end(), 0);
//    std::iota(index.begin(), index.begin() + instance.jobs + instance.machines, 0);
//
//    for (int row = 0; row < instance.machines; ++row) {
//        value[instance.jobs + instance.machines + row] = -1.0;
//        index[instance.jobs + instance.machines + row] = instance.jobs + row;
//    }
//
//    auto status1 = rmp_phase1->addCols(offset, ones.data(), zeros.data(), upper.data(), index.size(), start.data(), index.data(), value.data());
//    rmp_phase1->run();
//
//    // initialize RMP if empty
//cg_time.start();
//    if (rmp->getNumCol() == 0 && pricer_farkas.optimize(rmp_phase1->getSolution().row_dual, pricing, _reduced_costs) > -kHighsInf) {
//        //add_columns(_reduced_costs);
//
//        for (int m = 0; m < instance.machines; ++m) {
//            if (_reduced_costs[m] > 1e-6) {
//                auto& solution = pricing[m].solution;
//                std::stable_sort(solution.begin(), solution.end()); // ensure columns are sorted for faster search
//                rmp_phase1->addCol(0, 0, kHighsInf, static_cast<int>(pricing[m].solution.size()), pricing[m].solution.data(), _ones.data());
//            }
//        }
//
//        ++iteration_count;
//    }
//cg_time.pause();
//
//    // tight loop to restore feasibility (assuming possible!)
//    do {
//rmp_time.start();
//        auto status = rmp_phase1->run();
//		HighsModelStatus modelStatus = HighsModelStatus::kOptimal;
//
//        double count_infeasibilities = rmp_phase1->getObjectiveValue();
//
//        if (rmp_phase1->getObjectiveValue() > 1e-6) {
//            modelStatus = HighsModelStatus::kInfeasible;
//		}
//
//        // HiGHs has gotten into a bad state, so we need to reset?
//        if (status == HighsStatus::kError) {
//            std::cout << std::format("{}: Unrecoverable Error {} {}\n", instance.name, (int)status, (int)modelStatus);
//            return false;
//        }
//
//        has_dual_ray = modelStatus == HighsModelStatus::kInfeasible;
//        lp_iteration_count += rmp_phase1->getInfo().simplex_iteration_count;
//
//        if (modelStatus == HighsModelStatus::kUnknown && has_dual_ray == false) {
//			// there's an issue with dual simplex when tabooing pivots to prove infeasibility
//			// it returns unknown status with no dual ray
//            std::cout << std::format("{}: Error proving infeasibility and calculating dual ray - {}\n", instance.name, (int)modelStatus);
//            return false;
//        }
//rmp_time.pause();
//
//		// age the columns
//        const auto& solution = rmp_phase1->getSolution();
//        const HighsInt* basis = rmp_phase1->getBasicVariablesArray();
//        const uint32_t num_col = rmp_phase1->getNumCol();
//        uint32_t basis_size = 0;
//        column_management._age.resize(num_col - offset, iteration_count - 1);
//
//        for (uint32_t i = 0; i < rmp_phase1->getNumRow(); ++i) {
//            uint32_t idx = basis[i];
//
//            // include all basis columns, even if they have zero value
//            // this helps reduce number of simplex pivots
//            if (idx < num_col && idx > offset) {
//                bool non_zero = solution.col_value[idx] > 1e-6;
//                column_management._age[idx - offset] = iteration_count + non_zero;
//                basis_size += non_zero;
//            }
//        }
//
//cg_time.start();
//        if (has_dual_ray) {
//            pricer_farkas.update(rmp_phase1.get());
//
//            if (pricer_farkas.optimize(rmp_phase1->getSolution().row_dual, pricing, _reduced_costs) > -kHighsInf) {
//                //column_management.reduce(iteration_count); 
//                //add_columns(_reduced_costs);
//
//                for (int m = 0; m < instance.machines; ++m) {
//                    if (_reduced_costs[m] > 1e-6) {
//                        auto& solution = pricing[m].solution;
//                        std::stable_sort(solution.begin(), solution.end()); // ensure columns are sorted for faster search
//                        rmp_phase1->addCol(0, 0, kHighsInf, static_cast<int>(pricing[m].solution.size()), pricing[m].solution.data(), _ones.data());
//                    }
//                }
//            }
//            else {
//				std::cout << "Node infeasible!" << std::endl;
//                return false;
//            }
//        }
//cg_time.pause();
//
//        // debugging
//        if (iteration_count % ITERATION_OUTPUT == 0 && total_time.TotalSeconds() - previous_logging_time > ITERATION_TIME && has_dual_ray) {
//            tbl.output(iteration_count, _LB, "-", "-", "-", "-", "-", rmp_phase1->getNumCol() - instance.jobs - 2 * instance.machines, total_time.TotalSeconds(), lp_iteration_count, -count_infeasibilities);
//            previous_logging_time = total_time.TotalSeconds();
//        }
//
//        ++iteration_count;
//    } while (has_dual_ray == true && should_stop == false);
//
//	// add rmp_phase1 columns to rmp
//    for (size_t idx = offset, size = rmp_phase1->getNumCol(); idx < size; ++idx) {
//        const auto& col = get_column(*rmp_phase1.get(), idx);
//
//        auto end = std::prev(col.end());
//        auto machine = *end - instance.jobs;
//
//		// calculate cost
//		double cost = sum(col.begin(), end, 0.0, instance.profit[machine]);
//		rmp->addCol(cost, 0, kHighsInf, static_cast<int>(col.size()), (int*)col.data(), _ones.data());
//    }
//
//    rmp->run();
//
//    return true;
//}

//template <typename RmpSolver, template<typename> class FarkasType, template<typename> class PricerType>
//bool GapSolver<RmpSolver, FarkasType, PricerType>::restoreFeasibility(FarkasType<RmpSolver>& pricer_farkas) {
//    bool has_dual_ray = false;
//
//    // test a silly idea: reconstruct the RMP model with a phase I primal simplex to get dual ray
//    std::unique_ptr<RmpSolver> rmp_phase1;
//    rmp_phase1.reset(new RmpSolver);
//
//    rmp_phase1->setOptionValue("output_flag", false);
//    rmp_phase1->setOptionValue(kPresolveString, "off");
//    rmp_phase1->setOptionValue("random_seed", params.random_seed);
//    rmp_phase1->setOptionValue("threads", 1);
//    rmp_phase1->setOptionValue("simplex_strategy", "4");
//
//    auto model = SetCoverRestrictedProblem(instance.jobs, instance.machines, ObjSense::kMinimize);
//    rmp_phase1->passModel(model);
//
//    // modify with dummy variables, one per >= row, two per == row
//    // for set cover, # jobs >=, # machines == (so jobs+2xmachines)
//    std::vector<double> ones(instance.jobs + instance.machines, 1.0);
//    std::vector<double> zeros(instance.jobs + instance.machines, 0.0);
//    std::vector<double> upper(instance.jobs + instance.machines, kHighsInf);
//
//    // the first #jobs rows have one dummy variable each, the next #machines have two each (+ve and -ve)
//    std::vector<int> start(instance.jobs + instance.machines + 1);
//    std::vector<int> index(instance.jobs + instance.machines);
//    std::vector<double> value(instance.jobs + instance.machines, 1.0);
//
//    std::iota(start.begin(), start.end(), 0);
//    std::iota(index.begin(), index.end(), 0);
//
//    auto status1 = rmp_phase1->addCols(instance.jobs + instance.machines, ones.data(), zeros.data(), upper.data(), index.size(), start.data(), index.data(), value.data());
//    rmp_phase1->run();
//
//    // initialize RMP if empty
//    cg_time.start();
//    if (rmp->getNumCol() == 0 && pricer_farkas.optimize(rmp_phase1->getSolution().row_dual, pricing, _reduced_costs) > -kHighsInf) {
//        //add_columns(_reduced_costs);
//
//        for (int m = 0; m < instance.machines; ++m) {
//            if (_reduced_costs[m] > 1e-6) {
//                auto& solution = pricing[m].solution;
//                std::stable_sort(solution.begin(), solution.end()); // ensure columns are sorted for faster search
//                rmp_phase1->addCol(0, 0, kHighsInf, static_cast<int>(pricing[m].solution.size()), pricing[m].solution.data(), _ones.data());
//            }
//        }
//
//        ++iteration_count;
//    }
//    cg_time.pause();
//
//    // tight loop to restore feasibility (assuming possible!)
//    do {
//        rmp_time.start();
//        auto status = rmp_phase1->run();
//        HighsModelStatus modelStatus = HighsModelStatus::kOptimal;
//
//        double count_infeasibilities = rmp_phase1->getObjectiveValue();
//
//        if (rmp_phase1->getObjectiveValue() > 1e-6) {
//            modelStatus = HighsModelStatus::kInfeasible;
//        }
//
//        // HiGHs has gotten into a bad state, so we need to reset?
//        if (status == HighsStatus::kError) {
//            std::cout << std::format("{}: Unrecoverable Error {} {}\n", instance.name, (int)status, (int)modelStatus);
//            return false;
//        }
//
//        has_dual_ray = modelStatus == HighsModelStatus::kInfeasible;
//        lp_iteration_count += rmp_phase1->getInfo().simplex_iteration_count;
//
//        if (modelStatus == HighsModelStatus::kUnknown && has_dual_ray == false) {
//            // there's an issue with dual simplex when tabooing pivots to prove infeasibility
//            // it returns unknown status with no dual ray
//            std::cout << std::format("{}: Error proving infeasibility and calculating dual ray - {}\n", instance.name, (int)modelStatus);
//            return false;
//        }
//        rmp_time.pause();
//
//		// age the columns
//        const auto& solution = rmp_phase1->getSolution();
//        const HighsInt* basis = rmp_phase1->getBasicVariablesArray();
//        const uint32_t num_col = rmp_phase1->getNumCol();
//        uint32_t basis_size = 0;
//        column_management._age.resize(num_col - instance.jobs - instance.machines, iteration_count - 1);
//
//        for (uint32_t i = 0; i < rmp_phase1->getNumRow(); ++i) {
//            uint32_t idx = basis[i];
//
//            // include all basis columns, even if they have zero value
//            // this helps reduce number of simplex pivots
//            if (idx < num_col && idx > instance.jobs + instance.machines) {
//                bool non_zero = solution.col_value[idx] > 1e-6;
//                column_management._age[idx - instance.jobs - instance.machines] = iteration_count + non_zero;
//                basis_size += non_zero;
//            }
//        }
//
//        cg_time.start();
//        if (has_dual_ray) {
//			pricer_farkas.update(rmp_phase1.get());
//
//            if (pricer_farkas.optimize(solution.row_dual, pricing, _reduced_costs) > -kHighsInf) {
//                //column_management.reduce(iteration_count); 
//                //add_columns(_reduced_costs);
//
//                for (int m = 0; m < instance.machines; ++m) {
//                    if (_reduced_costs[m] > 1e-6) {
//                        auto& solution = pricing[m].solution;
//                        std::stable_sort(solution.begin(), solution.end()); // ensure columns are sorted for faster search
//                        rmp_phase1->addCol(0, 0, kHighsInf, static_cast<int>(pricing[m].solution.size()), pricing[m].solution.data(), _ones.data());
//                    }
//                }
//            }
//            else {
//                std::cout << "Node infeasible!" << std::endl;
//                return false;
//            }
//        }
//        cg_time.pause();
//
//        // debugging
//        if (iteration_count % ITERATION_OUTPUT == 0 && total_time.TotalSeconds() - previous_logging_time > ITERATION_TIME && has_dual_ray) {
//            tbl.output(iteration_count, _LB, "-", "-", "-", "-", "-", rmp_phase1->getNumCol() - instance.jobs - 2 * instance.machines, total_time.TotalSeconds(), lp_iteration_count, -count_infeasibilities);
//            previous_logging_time = total_time.TotalSeconds();
//        }
//
//        ++iteration_count;
//    } while (has_dual_ray == true);
//
//    // add rmp_phase1 columns to rmp
//    // all columns > jobs + 2 * machines
//    auto solution = rmp_phase1->getSolution();
//
//    for (size_t idx = instance.jobs + instance.machines, size = rmp_phase1->getNumCol(); idx < size; ++idx) {
//        const auto& col = get_column(*rmp_phase1.get(), idx);
//
//        auto end = std::prev(col.end());
//        auto machine = *end - instance.jobs;
//
//        // calculate cost
//        double cost = sum(col.begin(), end, 0.0, instance.profit[machine]);
//        rmp->addCol(cost, 0, kHighsInf, static_cast<int>(col.size()), (int*)col.data(), _ones.data());
//    }
//
//    rmp->run();
//
//    return true;
//}

//// partition version
//template <typename RmpSolver, template<typename> class FarkasType, template<typename> class PricerType>
//bool GapSolver<RmpSolver, FarkasType, PricerType>::restoreFeasibility(FarkasType<RmpSolver>& pricer_farkas) {
//    bool has_dual_ray = false;
//    int offset = 2 * (instance.jobs + instance.machines);
//
//    // test a silly idea: reconstruct the RMP model with a phase I primal simplex to get dual ray
//    std::unique_ptr<RmpSolver> rmp_phase1;
//    rmp_phase1.reset(new RmpSolver);
//
//    rmp_phase1->setOptionValue("output_flag", false);
//    rmp_phase1->setOptionValue(kPresolveString, "off");
//    rmp_phase1->setOptionValue("random_seed", params.random_seed);
//    rmp_phase1->setOptionValue("threads", 1);
//    rmp_phase1->setOptionValue("simplex_strategy", "4");
//
//    auto model = SetPartitionRestrictedProblem(instance.jobs, instance.machines, ObjSense::kMinimize);
//    rmp_phase1->passModel(model);
//
//    // modify with dummy variables, one per >= row, two per == row
//    // for set partition, # jobs ==, # machines ==, so 2x(jobs+machines)
//    std::vector<double> ones(offset, 1.0);
//    std::vector<double> zeros(offset, 0.0);
//    std::vector<double> upper(offset, kHighsInf);
//
//    // the first #jobs rows have one dummy variable each, the next #machines have two each (+ve and -ve)
//    std::vector<int> start(offset + 1);
//    std::vector<int> index(offset);
//    std::vector<double> value(offset, 1.0);
//
//    std::iota(start.begin(), start.end(), 0);
//	std::iota(index.begin(), index.begin() + instance.jobs + instance.machines, 0);
//
//    for (int row = 0; row < instance.jobs + instance.machines; ++row) {
//        value[instance.jobs + instance.machines + row] = -1.0;
//        index[instance.jobs + instance.machines + row] = row;
//    }
//
//    auto status1 = rmp_phase1->addCols(offset, ones.data(), zeros.data(), upper.data(), index.size(), start.data(), index.data(), value.data());
//    rmp_phase1->run();
//
//    // initialize RMP if empty
//    cg_time.start();
//    if (rmp->getNumCol() == 0 && pricer_farkas.optimize(rmp_phase1->getSolution().row_dual, pricing, _reduced_costs) > -kHighsInf) {
//        //add_columns(_reduced_costs);
//
//        for (int m = 0; m < instance.machines; ++m) {
//            if (_reduced_costs[m] > 1e-6) {
//                auto& solution = pricing[m].solution;
//                std::stable_sort(solution.begin(), solution.end()); // ensure columns are sorted for faster search
//                rmp_phase1->addCol(0, 0, kHighsInf, static_cast<int>(pricing[m].solution.size()), pricing[m].solution.data(), _ones.data());
//            }
//        }
//
//        ++iteration_count;
//    }
//    cg_time.pause();
//
//    // tight loop to restore feasibility (assuming possible!)
//    do {
//        rmp_time.start();
//        auto status = rmp_phase1->run();
//        HighsModelStatus modelStatus = HighsModelStatus::kOptimal;
//
//        double count_infeasibilities = rmp_phase1->getObjectiveValue();
//
//        if (rmp_phase1->getObjectiveValue() > 1e-6) {
//            modelStatus = HighsModelStatus::kInfeasible;
//        }
//
//        // HiGHs has gotten into a bad state, so we need to reset?
//        if (status == HighsStatus::kError) {
//            std::cout << std::format("{}: Unrecoverable Error {} {}\n", instance.name, (int)status, (int)modelStatus);
//            return false;
//        }
//
//        has_dual_ray = modelStatus == HighsModelStatus::kInfeasible;
//        lp_iteration_count += rmp_phase1->getInfo().simplex_iteration_count;
//
//        if (modelStatus == HighsModelStatus::kUnknown && has_dual_ray == false) {
//            // there's an issue with dual simplex when tabooing pivots to prove infeasibility
//            // it returns unknown status with no dual ray
//            std::cout << std::format("{}: Error proving infeasibility and calculating dual ray - {}\n", instance.name, (int)modelStatus);
//            return false;
//        }
//        rmp_time.pause();
//
//        // age the columns
//        const auto& solution = rmp_phase1->getSolution();
//        const HighsInt* basis = rmp_phase1->getBasicVariablesArray();
//        const uint32_t num_col = rmp_phase1->getNumCol();
//        uint32_t basis_size = 0;
//        column_management._age.resize(num_col - offset, iteration_count - 1);
//
//        for (uint32_t i = 0; i < rmp_phase1->getNumRow(); ++i) {
//            uint32_t idx = basis[i];
//
//            // include all basis columns, even if they have zero value
//            // this helps reduce number of simplex pivots
//            if (idx < num_col && idx > offset) {
//                bool non_zero = solution.col_value[idx] > 1e-6;
//                column_management._age[idx - offset] = iteration_count + non_zero;
//                basis_size += non_zero;
//            }
//        }
//
//        cg_time.start();
//        if (has_dual_ray) {
//            pricer_farkas.update(rmp_phase1.get());
//
//            if (pricer_farkas.optimize(rmp_phase1->getSolution().row_dual, pricing, _reduced_costs) > -kHighsInf) {
//                //column_management.reduce(iteration_count); 
//                //add_columns(_reduced_costs);
//
//                for (int m = 0; m < instance.machines; ++m) {
//                    if (_reduced_costs[m] > 1e-6) {
//                        auto& solution = pricing[m].solution;
//                        std::stable_sort(solution.begin(), solution.end()); // ensure columns are sorted for faster search
//                        rmp_phase1->addCol(0, 0, kHighsInf, static_cast<int>(pricing[m].solution.size()), pricing[m].solution.data(), _ones.data());
//                    }
//                }
//            }
//            else {
//                std::cout << "Node infeasible!" << std::endl;
//                return false;
//            }
//        }
//        cg_time.pause();
//
//        // debugging
//        if (iteration_count % ITERATION_OUTPUT == 0 && total_time.TotalSeconds() - previous_logging_time > ITERATION_TIME && has_dual_ray) {
//            tbl.output(iteration_count, _LB, "-", "-", "-", "-", "-", rmp_phase1->getNumCol() - instance.jobs - 2 * instance.machines, total_time.TotalSeconds(), lp_iteration_count, -count_infeasibilities);
//            previous_logging_time = total_time.TotalSeconds();
//        }
//
//        ++iteration_count;
//    } while (has_dual_ray == true && should_stop == false);
//
//    if (should_stop) {
//        return false;
//    }
//
//    // add rmp_phase1 columns to rmp
//    // all columns > jobs + 2 * machines
//    auto solution = rmp_phase1->getSolution();
//
//    for (size_t idx = offset, size = rmp_phase1->getNumCol(); idx < size; ++idx) {
//        //if (solution.col_value[idx] > 1e-6) {
//        const auto& col = get_column(*rmp_phase1.get(), idx);
//
//        auto end = std::prev(col.end());
//        auto machine = *end - instance.jobs;
//
//        // calculate cost
//        double cost = sum(col.begin(), end, 0.0, instance.profit[machine]);
//        rmp->addCol(cost, 0, kHighsInf, static_cast<int>(col.size()), (int*)col.data(), _ones.data());
//        //}
//    }
//
//    rmp->run();
//
//    return true;
//}

template <typename RmpSolver, template<typename> class FarkasType, template<typename> class PricerType>
void GapSolver<RmpSolver, FarkasType, PricerType>::updateCompactSolution() {
    const auto& solution = rmp->getSolution();

    if (solution.value_valid) {
        basis_size = 0;
        std::fill(_compact_solution.begin(), _compact_solution.end(), 0);

        // try to find UB solution
        for (size_t idx = 0, size = rmp->getNumCol(); idx < size; ++idx) {
            if (solution.col_value[idx] > 1e-6) {
                ++basis_size;
				const auto& col = get_column(*rmp.get(), idx);

                auto end = std::prev(col.end());
                auto machine = *end - instance.jobs;

                for (auto it = col.begin(); it != end; ++it) {
                    _compact_solution[machine * instance.jobs + *it] += solution.col_value[idx];
                }
            }
        }

        // check if integral
        fractional_count = 0;

        for (double val : _compact_solution) {
            fractional_count += size_t(val > 1e-6 && val < 1 - 1e-6);
        }

        if (fractional_count == 0) {
            double tmpUB = 0;
            for (int m = 0; m < instance.machines; ++m) {
                auto& profit = instance.profit[m];
                for (int j = 0; j < instance.jobs; ++j) {
                    tmpUB += profit[j] * std::ceil(_compact_solution[m * instance.jobs + j] - 1e-6);
                }
            }

            if (_UB > tmpUB) {
                _UB = tmpUB;
            }
        }

        // if using cover (instead of partition), we might need to remove duplicate jobs
        if (basis_size == instance.machines) {
            double tmpUB = remove_duplicates();

            if (_UB > tmpUB) {
                _UB = tmpUB;
            }
        }
    }
}

// we might have duplicate jobs since RMP uses cover (instead of partition)
// if we have an integral solution we need to remove duplicates to get the correct UB
template <typename RmpSolver, template<typename> class FarkasType, template<typename> class PricerType>
double GapSolver<RmpSolver, FarkasType, PricerType>::remove_duplicates() {
    const auto& solution = rmp->getSolution();
    const HighsInt* basis = rmp->getBasicVariablesArray();
    const HighsInt num_col = rmp->getNumCol();
	const int basis_size = rmp->getNumRow();

    // assume unlikely to have duplicates
	bool has_duplicates = false;
    std::vector<bool> has_job(instance.jobs, false);

    for (int i = 0; i < basis_size; ++i) {
        int idx = basis[i];

        if (idx < num_col && solution.col_value[idx] > 1e-6) {
			const auto& col = get_column(*rmp, idx);

            for (auto it = col.begin(), end = std::prev(col.end()); it != end; ++it) {
				has_duplicates |= has_job[*it];
                has_job[*it] = true;
            }
        }
    }

	// no duplicates
	if (!has_duplicates) {
		return rmp->getObjectiveValue();
	}
    else {
        // assume that basis = machines, then solution (and duplicate) is integral
        double total_cost = rmp->getObjectiveValue();
        std::vector<std::vector<int>> machine_jobs(instance.machines);
		std::vector<std::vector<int>> job_machines(instance.jobs);

		// copy job/machine for easy lookup
        for (int i = rmp->getNumRow() - 1; i >= 0; --i) {
            if (basis[i] < num_col && solution.col_value[basis[i]] > 1e-6) {
				const auto& col = get_column(*rmp, basis[i]);
				auto end = std::prev(col.end());
				auto machine = *end - instance.jobs;
				std::copy(col.begin(), end, std::back_inserter(machine_jobs[machine]));

                for (auto j : machine_jobs[machine]) {
                    job_machines[j].emplace_back(machine);
                }
            }
        }

        std::vector<bool> updated_machines(instance.machines);

        // choose cheapest machine for each duplicate job
		for (int j = 0; j < instance.jobs; ++j) {
            if (job_machines[j].size() > 1) {
				int best_machine = -1;
				double min_cost = std::numeric_limits<double>::max();
                
				for (int m : job_machines[j]) {
					double cost = instance.profit[m][j];
					total_cost -= cost;

					if (min_cost > cost) {
                        if (best_machine != -1) {
                            // remove from previous machine
                            auto it = std::lower_bound(machine_jobs[best_machine].begin(), machine_jobs[best_machine].end(), j);
                            machine_jobs[best_machine].erase(it);
							updated_machines[best_machine] = true;
                        }

						// update best machine
                        min_cost = cost;
                        best_machine = m;
					}
                    else {
                        // remove from machine
                        auto it = std::lower_bound(machine_jobs[m].begin(), machine_jobs[m].end(), j);
                        machine_jobs[m].erase(it);
                        updated_machines[m] = true;
                    }
				}

                total_cost += instance.profit[best_machine][j];
            }
		}

        // add modified columns back into RMP to get partition instead of cover
        // need to do this for each of the machines
        for (int m = 0; m < instance.machines; ++m) {
            if (updated_machines[m]) {
                double cost = sum(machine_jobs[m], 0.0, instance.profit[m]);
                machine_jobs[m].push_back(instance.jobs + m);
                rmp->addCol(cost, 0, kHighsInf, static_cast<int>(machine_jobs[m].size()), machine_jobs[m].data(), _ones.data());
            }
        }

        rmp->run();
		return total_cost;
    }
}

template <typename RmpSolver, template<typename> class FarkasType, template<typename> class PricerType>
int GapSolver<RmpSolver, FarkasType, PricerType>::add_columns(std::vector<double>& reduced_costs) {
    int count = 0;

    for (int m = 0; m < instance.machines; ++m) {
        auto& profit = instance.profit[m];

        if (reduced_costs[m] > 1e-6) {
			auto& solution = pricing[m].solution;
            std::stable_sort(solution.begin(), solution.end()); // ensure columns are sorted for faster search
            double cost = sum(solution.begin(), --solution.end(), 0.0, profit);

            rmp->addCol(cost, 0, kHighsInf, static_cast<int>(pricing[m].solution.size()), pricing[m].solution.data(), _ones.data());
            ++count;
        }
    }

    return count;
}

//
//template <typename RmpSolver, template<typename> class FarkasType, template<typename> class PricerType>
//bool GapSolver<RmpSolver, FarkasType, PricerType>::restoreFeasibility(FarkasType<RmpSolver>& pricer_farkas) {
//    bool has_dual_ray = false;
//    std::vector<double> dual_ray(rmp->getNumRow(), 1);
//
//    // test a silly idea: reconstruct the RMP model with a phase I primal simplex to get dual ray
//    std::unique_ptr<RmpSolver> rmp_phase1;
//    rmp_phase1.reset(new RmpSolver);
//
//    rmp_phase1->setOptionValue("output_flag", false);
//    rmp_phase1->setOptionValue(kPresolveString, "off");
//    rmp_phase1->setOptionValue("random_seed", params.random_seed);
//    rmp_phase1->setOptionValue("threads", 1);
//    rmp_phase1->setOptionValue("simplex_strategy", "4");
//
//    auto model = SetCoverRestrictedProblem(instance.jobs, instance.machines, ObjSense::kMinimize);
//    rmp_phase1->passModel(model);
//
//    // modify with dummy variables, one per >= row, two per == row
//    // for set cover, # jobs >=, # machines == (so jobs+2xmachines)
//    std::vector<int> dummy(instance.jobs + 2 * instance.machines);
//    std::vector<double> ones(instance.jobs + 2 * instance.machines, 1.0);
//    std::vector<double> zeros(instance.jobs + 2 * instance.machines, 0.0);
//    std::vector<double> upper(instance.jobs + 2 * instance.machines, kHighsInf);
//
//    // the first #jobs rows have one dummy variable each, the next #machines have two each (+ve and -ve)
//    std::vector<int> start(instance.jobs + 2 * instance.machines + 1);
//    std::vector<int> index(instance.jobs + 2 * instance.machines);
//    std::vector<double> value(instance.jobs + 2 * instance.machines, 1.0);
//
//    std::iota(dummy.begin(), dummy.end(), 0);
//    std::iota(start.begin(), start.end(), 0);
//    std::iota(index.begin(), index.begin() + instance.jobs, 0);
//
//    for (int row = 0; row < instance.machines; ++row) {
//        value[instance.jobs + 2 * row + 1] = -1.0;
//
//        index[instance.jobs + 2 * row + 0] = instance.jobs + row;
//        index[instance.jobs + 2 * row + 1] = instance.jobs + row;
//    }
//
//    auto status1 = rmp_phase1->addCols(dummy.size(), ones.data(), zeros.data(), upper.data(), index.size(), start.data(), index.data(), value.data());
//    rmp_phase1->run();
//
//    auto test = rmp_phase1->getSolution();
//    dual_ray = test.row_dual;
//    //if constexpr (std::is_same<RmpSolver, Highs>::value) {
//    //    rmp->setOptionValue("simplex_strategy", "1"); // need to use dual solver for extreme ray
//    //}
//    //else {
//    //    rmp->setOptionValue("simplex_strategy", "4"); // use primal simplex for other solvers
//    //}
//
//    //rmp->setOptionValue("allow_unbounded_or_infeasible", true);
//
//    // initialize RMP if empty
//    cg_time.start();
//    if (rmp->getNumCol() == 0 && pricer_farkas.optimize(dual_ray, pricing, _reduced_costs) > -kHighsInf) {
//        //add_columns(_reduced_costs);
//
//        for (int m = 0; m < instance.machines; ++m) {
//            if (_reduced_costs[m] > 1e-6) {
//                auto& solution = pricing[m].solution;
//                std::stable_sort(solution.begin(), solution.end()); // ensure columns are sorted for faster search
//                rmp_phase1->addCol(0, 0, kHighsInf, static_cast<int>(pricing[m].solution.size()), pricing[m].solution.data(), _ones.data());
//            }
//        }
//
//        ++iteration_count;
//    }
//    cg_time.pause();
//
//    // tight loop to restore feasibility (assuming possible!)
//    do {
//        rmp_time.start();
//        auto status = rmp_phase1->run();
//        HighsModelStatus modelStatus = HighsModelStatus::kOptimal;
//
//        double count_infeasibilities = rmp_phase1->getObjectiveValue();
//
//        if (rmp_phase1->getObjectiveValue() > 1e-6) {
//            modelStatus = HighsModelStatus::kInfeasible;
//        }
//
//        has_dual_ray = modelStatus == HighsModelStatus::kInfeasible;
//        //dual_ray = rmp_phase1->getSolution().row_dual;
//
//        //auto status = rmp->run();
//        //auto modelStatus = rmp->getModelStatus();
//
//        // HiGHs has gotten into a bad state, so we need to reset?
//        if (status == HighsStatus::kError) {
//            std::cout << std::format("{}: Unrecoverable Error {} {}\n", instance.name, (int)status, (int)modelStatus);
//            return false;
//        }
//
//        //// checks if we have a dual ray
//        //// i.e. y^T A <= 0, y^T b > 0
//  //      auto is_farkas_certificate = [](const HighsLp& lp, const std::vector<double>& duals) {
//        //	const auto& A = lp.a_matrix_;
//        //	const HighsInt num_row = lp.num_row_;
//        //	const HighsInt num_col = lp.num_col_;
//        //	double ray_value = 0.0;
//        //	
//  //          // matrix is column-wise
//        //	for (HighsInt col = 0; col < num_col; ++col) {
//  //              double value = 0.0;
//
//        //		for (HighsInt idx = A.start_[col]; idx < A.start_[col + 1]; ++idx) {
//        //			HighsInt row = A.index_[idx];
//        //			value += A.value_[idx] * duals[row];
//        //		}
//
//  //              if (value > 1e-6) {
//  //                  return false;
//  //              }
//        //	}
//
//  //          for (HighsInt row = 0; row < num_row; ++row) {
//  //              double b_value = 0.0;
//
//        //		// positive duals for lower bounds
//  //              if (duals[row] > -1e-6 && lp.row_lower_[row] > -kHighsInf) {
//  //                  b_value = lp.row_lower_[row];
//  //              }
//
//  //              // negative duals for upper bounds
//  //              if (duals[row] < 1e-6 && lp.row_upper_[row] < kHighsInf) {
//  //                  b_value = -lp.row_upper_[row];
//  //              }
//  //              ray_value += duals[row] * b_value;
//        //	}
//
//        //	if (ray_value < 1e-6) {
//        //		return false;
//        //	}
//
//  //          return true;
//        //};
//
//  //      has_dual_ray = false;
//
//  //      if (modelStatus == HighsModelStatus::kInfeasible) {
//  //          // constexpr if RmpSolver is Highs
//  //          if constexpr (std::is_same<RmpSolver, Highs>::value) {
//  //              // check to see if the solution yields a dual ray
//        //		Highs* highs_ptr = static_cast<Highs*>(rmp.get());
//  //              auto solution = highs_ptr->getSolution();
//
//  //              // check all rows for feasibility
//  //              const auto& lp = highs_ptr->getLp();
//
//  //              const auto& A = lp.a_matrix_;
//  //              
//  //              const HighsInt num_row = lp.num_row_;
//  //              const HighsInt num_col = lp.num_col_;
//  //              
//  //              int count_infeasibilities = 0;
//
//        //		// matrix is column-wise
//        //		std::vector<double> row_values(num_row, 0.0);
//        //		std::vector<double> col_values(num_col, 0.0);
//  //              std::vector<double> norm_duals = solution.row_dual;
//
//  //              for (HighsInt col = 0; col < num_col; ++col) {
//  //                  for (HighsInt idx = A.start_[col]; idx < A.start_[col + 1]; ++idx) {
//  //                      HighsInt row = A.index_[idx];
//  //                      row_values[row] += A.value_[idx] * solution.col_value[col];
//  //                  }
//        //		}
//
//  //              double max_dual = 0.0;
//        //		double sum_values = 0.0;
//
//  //              for (HighsInt i = 0; i < num_row; ++i) {
//  //                  if (lp.row_lower_[i] > -kHighsInf && row_values[i] < lp.row_lower_[i] - 1e-6) {
//  //                      ++count_infeasibilities;
//  //                  }
//
//  //                  if (lp.row_upper_[i] < kHighsInf && row_values[i] > lp.row_upper_[i] + 1e-6) {
//  //                      ++count_infeasibilities;
//  //                  }
//
//        //			row_values[i] = std::max(lp.row_lower_[i] - row_values[i], std::min(lp.row_upper_[i] - row_values[i], 0.0)); // reset for dual ray
//        //			//max_dual = std::max(max_dual, std::abs(norm_duals[i]));
//  //   //               sum_values += std::abs(row_values[i]);
//  //              }
//
//        //		//// normalize dual ray
//  //  //            if (max_dual > 1e-6) {
//  //  //                for (HighsInt i = 0; i < num_row; ++i) {
//  //  //                    norm_duals[i] /= max_dual;
//  //  //                }
//        //		//}
//
//  //              // basis inverse
//  //              // can I reproduce the dual values from the inverse basis?
//
//  //              std::vector<double> test_ray(num_row, 0.0);
//
//  //              const HighsInt* basic_ptr = highs_ptr->getBasicVariablesArray();
//        //		std::vector<HighsInt> basic_variables(basic_ptr, basic_ptr + num_row);
//        //		std::vector<double> cbasis(num_row, 0.0);
//
//  //    //          for (int basis_col = 0; basis_col < num_row; ++basis_col) {
//  //    //              if (basic_ptr[basis_col] < num_col) {
//        //				//cbasis[basis_col] = lp.col_cost_[basic_ptr[basis_col]];
//  //    //              }
//  //    //          }
//
//        //		// testing idea about phase I basis solve
//  //              for (int basis_col = 0; basis_col < num_row; ++basis_col) {
//  //                  // if basis is a row, then cost is 1
//  //                  if (basic_ptr[basis_col] >= num_col) {
//        //				double tmp = row_values[basic_ptr[basis_col] - num_col];
//  //                      cbasis[basis_col] = std::abs(tmp) > 1e-6 ? (tmp > 0 ? 1 : -1) : 0;
//  //                  }
//  //              }
//
//        //		std::vector<double> row_solution(num_row, 0.0);
//  //              std::vector<HighsInt> solution_idx(num_row, 0);
//        //		HighsInt solution_nnz = 0;
//        //		highs_ptr->getBasisTransposeSolve(cbasis.data(), row_solution.data(), &solution_nnz, solution_idx.data());
//
//  //              bool check1 = is_farkas_certificate(lp, row_values);
//
//        //		bool check = is_farkas_certificate(lp, row_solution);
//
//  //              bool check3 = is_farkas_certificate(lp, solution.row_dual);
//  //              dual_ray = row_solution;
//
//  //              for (int row = 0; row < num_row; ++row) {
//        //			std::vector<double> tmp(num_row, 0.0);
//  //                  int nnz = 0;
//        //			std::vector<HighsInt> idx(num_row, 0);
//
//        //			highs_ptr->getBasisInverseRow(row, tmp.data(), &nnz, idx.data());
//
//  //                  std::vector<double> tmp2(num_row, 0.0);
//  //                  for (int i = 0; i < nnz; ++i) {
//  //                      tmp2[idx[i]] = -tmp[i];
//  //                  }
//
//  //                  if (is_farkas_certificate(lp, tmp2)) {
//        //				dual_ray = tmp2;
//  //                      break;
//  //                  }
//        //		}
//
//        //		has_dual_ray = (count_infeasibilities > 0);
//        //		//dual_ray = norm_duals;
//  //          }
//  //          else {
//  //              // nothing to do, Gurobi already provides dual ray
//  //              rmp->getDualRay(has_dual_ray, dual_ray.data());
//  //          }
//        //}
//
//        //has_dual_ray = modelStatus == HighsModelStatus::kInfeasible;
//        //dual_ray = rmp->getSolution().row_dual;
//
//  //      rmp->getDualRay(has_dual_ray, dual_ray.data());
//  //      lp_iteration_count += rmp->getInfo().simplex_iteration_count;
//
//        //int count_infeasibilities = 0;
//
//  //      if (modelStatus == HighsModelStatus::kInfeasible) {
//  //          auto solution = rmp->getSolution();
//
//  //          for (int i = 0; i < rmp->getNumRow(); ++i) {
//  //              if (solution.row_value[i] < 1 - 1e-7) {
//  //                  ++count_infeasibilities;
//  //              }
//
//        //		// if set cover
//  //              if (i >= instance.jobs && solution.row_value[i] > 1 + 1e-7) {
//  //                  ++count_infeasibilities;
//  //              }
//
//  //              //// if set partition
//  //              //if (solution.row_value[i] > 1 + 1e-7) {
//  //              //    ++count_infeasibilities;
//  //              //}
//  //          }
//  //      }
//
//        lp_iteration_count += rmp_phase1->getInfo().simplex_iteration_count;
//
//        if (modelStatus == HighsModelStatus::kUnknown && has_dual_ray == false) {
//            // there's an issue with dual simplex when tabooing pivots to prove infeasibility
//            // it returns unknown status with no dual ray
//            std::cout << std::format("{}: Error proving infeasibility and calculating dual ray - {}\n", instance.name, (int)modelStatus);
//            return false;
//        }
//        rmp_time.pause();
//
//        cg_time.start();
//        if (has_dual_ray) {
//            if (pricer_farkas.optimize(rmp_phase1->getSolution().row_dual, pricing, _reduced_costs) > -kHighsInf) {
//                //column_management.reduce(iteration_count); 
//                //add_columns(_reduced_costs);
//
//                for (int m = 0; m < instance.machines; ++m) {
//                    if (_reduced_costs[m] > 1e-6) {
//                        auto& solution = pricing[m].solution;
//                        std::stable_sort(solution.begin(), solution.end()); // ensure columns are sorted for faster search
//                        rmp_phase1->addCol(0, 0, kHighsInf, static_cast<int>(pricing[m].solution.size()), pricing[m].solution.data(), _ones.data());
//                    }
//                }
//            }
//            else {
//                std::cout << "Node infeasible!" << std::endl;
//                return false;
//            }
//        }
//        cg_time.pause();
//
//        // debugging
//        if (iteration_count % ITERATION_OUTPUT == 0 && total_time.TotalSeconds() - previous_logging_time > ITERATION_TIME && has_dual_ray) {
//            tbl.output(iteration_count, _LB, "-", "-", "-", "-", "-", rmp_phase1->getNumCol() - instance.jobs - 2 * instance.machines, total_time.TotalSeconds(), lp_iteration_count, -count_infeasibilities);
//            previous_logging_time = total_time.TotalSeconds();
//        }
//
//        ++iteration_count;
//    } while (has_dual_ray == true);
//
//    // add rmp_phase1 columns to rmp
//    // all columns > jobs + 2 * machines
//    auto solution = rmp_phase1->getSolution();
//
//    for (size_t idx = instance.jobs + 2 * instance.machines, size = rmp_phase1->getNumCol(); idx < size; ++idx) {
//        //if (solution.col_value[idx] > 1e-6) {
//        const auto& col = get_column(*rmp_phase1.get(), idx);
//
//        auto end = std::prev(col.end());
//        auto machine = *end - instance.jobs;
//
//        // calculate cost
//        double cost = sum(col.begin(), end, 0.0, instance.profit[machine]);
//        rmp->addCol(cost, 0, kHighsInf, static_cast<int>(col.size()), (int*)col.data(), _ones.data());
//        //}
//    }
//
//    rmp->run();
//
//    return true;
//}