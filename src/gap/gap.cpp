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

// Supported template instantiations
template class DantzigPrice<Highs>;

template class DantzigFarkas<Highs>;
template class LagrangeTemplateFarkas<Highs>;
template class TemplateFarkas<Highs>;

template int GapSolver<Highs>::solve(DantzigPrice<Highs>& pricer, DantzigFarkas<Highs>& pricer_farkas);
template int GapSolver<Highs>::solve(WentgesPrice<Highs>& pricer, DantzigFarkas<Highs>& pricer_farkas);
template int GapSolver<Highs>::solve(TemplatePrice<Highs>& pricer, TemplateFarkas<Highs>& pricer_farkas);
template int GapSolver<Highs>::solve(LagrangeTemplatePrice<Highs>& pricer, LagrangeTemplateFarkas<Highs>& pricer_farkas);

template bool GapSolver<Highs>::restoreFeasibility(DantzigFarkas<Highs>& pricer_farkas);
template bool GapSolver<Highs>::restoreFeasibility(TemplateFarkas<Highs>& pricer_farkas);

#ifdef SUPPORT_GUROBI

template class DantzigPrice<GurobiHighs>;

template class DantzigFarkas<GurobiHighs>;
template class LagrangeTemplateFarkas<GurobiHighs>;
template class TemplateFarkas<GurobiHighs>;

template int GapSolver<GurobiHighs>::solve(DantzigPrice<GurobiHighs>& pricer, DantzigFarkas<GurobiHighs>& pricer_farkas);
template int GapSolver<GurobiHighs>::solve(WentgesPrice<GurobiHighs>& pricer, DantzigFarkas<GurobiHighs>& pricer_farkas);
template int GapSolver<GurobiHighs>::solve(TemplatePrice<GurobiHighs>& pricer, TemplateFarkas<GurobiHighs>& pricer_farkas);
template int GapSolver<GurobiHighs>::solve(LagrangeTemplatePrice<GurobiHighs>& pricer, LagrangeTemplateFarkas<GurobiHighs>& pricer_farkas);

template bool GapSolver<GurobiHighs>::restoreFeasibility(DantzigFarkas<GurobiHighs>& pricer_farkas);
template bool GapSolver<GurobiHighs>::restoreFeasibility(TemplateFarkas<GurobiHighs>& pricer_farkas);

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

template <typename RmpSolver>
void GapSolver<RmpSolver>::presolve() {
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

template <typename RmpSolver>
template <typename PricerType, typename FarkasPricerType>
int GapSolver<RmpSolver>::solve(PricerType& pricer, FarkasPricerType& pricer_farkas) {
    highs::parallel::initialize_scheduler(1);
    total_time.start();
    tbl.write_header();

    presolve();

	if (_UB == _LB) {
        std::cout << std::format("Compact solution is integer optimal.");
        return 0;
	}

    bool should_stop = false;
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
    std::function<HighsInt()> get_lp_iters = [&]() { return rmp->getInfo().simplex_iteration_count; };

    auto model = SetCoverRestrictedProblem(instance.jobs, instance.machines, ObjSense::kMinimize);
    rmp->passModel(model);
    pricer.init(&_executor, rmp.get(), &instance);

    // initialize pricing
    pricing.init(instance);
    pricer_farkas.init(&_executor, pricer, pricing, lp);

    double optimal_pricing = 0.0;
	int added_columns = 1; // initialize to 1 to avoid NaN in first iteration

    if (!restoreFeasibility(pricer_farkas))
        return -1;

    double _rmpLB = rmp->getObjectiveValue();
    updateCompactSolution();
	std::string params_json = params.to_json();

    tbl.output(iteration_count, _LB, _UB, "-", _rmpLB, "-", basis_size, rmp->getNumCol(), total_time.TotalSeconds(), lp_iteration_count, fractional_count);
	csv_writer.append_row(instance.name, instance.name[0], instance.machines, instance.jobs, pricer.name, params.solver, params.column_retention, params.replication, iteration_count,
        _LB, _UB, "", _rmpLB, "", basis_size, rmp->getNumCol(), rmp_time.TotalSeconds(), cg_time.TotalSeconds(), total_time.TotalSeconds(), 
        lp_iteration_count, lp_iteration_count / rmp_time.TotalSeconds(), lp_iteration_count / static_cast<double>(rmp->getNumCol()), int(basis_size == instance.machines), int(false), params_json, -1);

    // primal simplex for warm-start "add columns"
    rmp->setOptionValue("simplex_strategy", "4");
    rmp->setOptionValue("allow_unbounded_or_infeasible", false);  // not sure if this adds unnecessary overheads
    //rmp->setOptionValue(kSolverString, kIpmString);
    //rmp->setOptionValue(kRunCrossoverString, kHighsOffString);
    pricer.init_feasible();
    column_management.init(rmp.get(), &instance, params);

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
            csv_writer.append_row(instance.name, instance.name[0], instance.machines, instance.jobs, pricer.name, params.solver, params.column_retention, params.replication, iteration_count,
                _LB, _UB, gap * 100, _rmpLB, optimal_pricing, basis_size, rmp->getNumCol(), rmp_time.TotalSeconds(), cg_time.TotalSeconds(), total_time.TotalSeconds(), 
                lp_iteration_count, lp_iteration_count / rmp_time.TotalSeconds(), lp_iteration_per_column, int(basis_size==instance.machines), int(false), "", 0);

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
    tbl.output(iteration_count, lb, _UB, gap, _rmpLB, optimal_pricing, basis_size, rmp->getNumCol(), total_time.TotalSeconds(), lp_iteration_count, fractional_count);

	// provide additional status information
	int last_status = 0;

    if (gap < params.gap || lb + 1e-6 >= _rmpLB)
        last_status = 2; // gap
    else if (added_columns == 0)
        last_status = 3; // optimal
    else if (params.time_limit > 0 && total_time.TotalSeconds() > params.time_limit)
        last_status = -4; // time limit
    else if (should_stop)
		last_status = -5; // user interrupt

    csv_writer.append_row(instance.name, instance.name[0], instance.machines, instance.jobs, pricer.name, params.solver, params.column_retention, params.replication, iteration_count,
        _LB, _UB, std::abs(gap*100), _rmpLB, optimal_pricing, basis_size, rmp->getNumCol(), rmp_time.TotalSeconds(), cg_time.TotalSeconds(), total_time.TotalSeconds(),
        lp_iteration_count, lp_iteration_count / rmp_time.TotalSeconds(), avg_pivots_per_column, int(!std::isnan(gap)), int((params.time_limit > 0 && total_time.TotalSeconds() > params.time_limit)), params_json, last_status);

	// final entry only if not user interrupted
    if (!should_stop) {
        csv_writer.append_row(instance.name, instance.name[0], instance.machines, instance.jobs, pricer.name, params.solver, params.column_retention, params.replication, iteration_count,
            _LB, _UB, std::abs(gap * 100), _rmpLB, optimal_pricing, basis_size, rmp->getNumCol(), rmp_time.TotalSeconds(), cg_time.TotalSeconds(), total_time.TotalSeconds(),
            lp_iteration_count, lp_iteration_count / rmp_time.TotalSeconds(), avg_pivots_per_column, int(!std::isnan(gap)), int((params.time_limit > 0 && total_time.TotalSeconds() > params.time_limit)), params_json, 1);
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

template <typename RmpSolver>
template <typename FarkasPricerType>
bool GapSolver<RmpSolver>::restoreFeasibility(FarkasPricerType &pricer_farkas) {
    bool has_dual_ray = false;
    std::vector<double> dual_ray(rmp->getNumRow(), 1);

    rmp->setOptionValue("simplex_strategy", "1"); // need to use dual solver for extreme ray
    rmp->setOptionValue("allow_unbounded_or_infeasible", true);

    // initialize RMP if empty
cg_time.start();
    if (rmp->getNumCol() == 0 && pricer_farkas.optimize(dual_ray, pricing, _reduced_costs) > -kHighsInf) {
        add_columns(_reduced_costs);
        ++iteration_count;
    }
cg_time.pause();

    // tight loop to restore feasibility (assuming possible!)
    do {
rmp_time.start();
        auto status = rmp->run();

        // HiGHs has gotten into a bad state, so we need to reset?
        if (status == HighsStatus::kError) {
            std::cout << std::format("{}: Unrecoverable Error\n", instance.name);
            return false;
        }

        rmp->getDualRay(has_dual_ray, dual_ray.data());
        lp_iteration_count += rmp->getInfo().simplex_iteration_count;

        if (rmp->getModelStatus() == HighsModelStatus::kUnknown && has_dual_ray == false) {
            std::cout << std::format("{}: Test Error - {}\n", instance.name, (int)rmp->getModelStatus());
            return false;
        }
rmp_time.pause();

cg_time.start();
        if (has_dual_ray) {
            if (pricer_farkas.optimize(dual_ray, pricing, _reduced_costs) > -kHighsInf) {
                add_columns(_reduced_costs);
            }
            else {
				std::cout << "Node infeasible!" << std::endl;
                return false;
            }
        }
cg_time.pause();

        // debugging
        if (iteration_count % ITERATION_OUTPUT == 0 && total_time.TotalSeconds() - previous_logging_time > ITERATION_TIME && has_dual_ray) {
            tbl.output(iteration_count, _LB, "-", "-", "-", "-", "-", rmp->getNumCol(), total_time.TotalSeconds(), lp_iteration_count, fractional_count);
            previous_logging_time = total_time.TotalSeconds();
        }

        ++iteration_count;
    } while (has_dual_ray == true);

    return true;
}

template <typename RmpSolver>
void GapSolver<RmpSolver>::updateCompactSolution() {
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
template <typename RmpSolver>
double GapSolver<RmpSolver>::remove_duplicates() {
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

template <typename RmpSolver>
int GapSolver<RmpSolver>::add_columns(std::vector<double>& reduced_costs) {
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

