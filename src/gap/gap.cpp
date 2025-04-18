#include <vector>
#include <algorithm>
#include "gap.h"
#include "block/column_generation.h"
#include "highs/util/HighsIntegers.h"
#include <numeric>

#include "gap_instance.h"
#include <filesystem>
#include "quill/bundled/fmt/format.h"
#include "utils.h"

// Supported template instantiations
template void GapSolver::solve(TemplatePrice& pricer, TemplateFarkas& pricer_farkas);
template void GapSolver::solve(FixedTemplatePrice& pricer, FixedTemplateFarkas& pricer_farkas);
template void GapSolver::solve(DantzigPrice& pricer, DantzigFarkas& pricer_farkas);
template void GapSolver::solve(WentgesPrice& pricer, DantzigFarkas& pricer_farkas);
template void GapSolver::solve(WentgesTemplatePrice& pricer, WentgesTemplateFarkas& pricer_farkas);
template void GapSolver::solve(WentgesTemplatePrice& pricer, TemplateFarkas& pricer_farkas);

template bool GapSolver::restoreFeasibility(DantzigFarkas& pricer_farkas);
template bool GapSolver::restoreFeasibility(TemplateFarkas& pricer_farkas);

double remove_duplicates(GapInstance& instance, Highs* rmp);
bool add_columns(Highs* rmp, GapInstance& instance, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs, std::vector<double>& ones);

template <>
void TemplateFarkas::init<FixedTemplatePrice>(FixedTemplatePrice& pricer, PricingBlockVector<GapPricing>& pricing, gap_compact& lp) {
    _instance = pricer._instance;
    _template = pricer._template;
}

void DualColumnManagement::reduce(size_t iteration_count) {
    const int PREVIOUS_FACTOR = 3; // at least x basis kept columns should be those generated in the "last iterations"
    const int MAX_FACTOR = 6;
	const int MIN_COL_REDUCE = 8000;
    const int MIN_REMAINING = 4000;

    if (_rmp->getNumCol() > MIN_COL_REDUCE && _rmp->getNumCol() > MAX_FACTOR * _rmp->getNumRow()) {
        std::vector<double> reduced_costs(_rmp->getNumCol(), 0);

        auto& solution = _rmp->getSolution();
        calculate_reduced_costs(*_rmp, solution.row_dual, reduced_costs);

        const HighsInt* basis = _rmp->getBasicVariablesArray();
        const HighsInt num_col = _rmp->getNumCol();

        for (int i = _rmp->getNumRow() - 1; i >= 0; --i) {
            if (basis[i] < num_col) {
                reduced_costs[basis[i]] = kHighsInf;
            }
        }

        // sort by reduced cost, take lowest values such that remaining # cols = 3 * # rows
        std::vector<int> sorted_indices(_rmp->getNumCol());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::stable_sort(sorted_indices.begin(), sorted_indices.end(), [&](int a, int b) { return reduced_costs[a] < reduced_costs[b]; });

        // remove basis columns (at the end of the sorted list)
        int keep = std::max(MIN_REMAINING, PREVIOUS_FACTOR * _rmp->getNumRow());
        int size_to_remove = std::max(0, int(sorted_indices.size()) - 1 - keep);

        // shouldn't happen, but just in case - don't remove the basis columns
        while (size_to_remove > 0 && reduced_costs[sorted_indices[size_to_remove]] == kHighsInf) {
            --size_to_remove;
        }

        // want to keep PREVIOUS_FACTOR * # rows columns, i.e., remove the rest
		sorted_indices.resize(size_to_remove);
        std::sort(sorted_indices.begin(), sorted_indices.end());

        // remove the columns!
        _rmp->deleteCols(sorted_indices.size(), sorted_indices.data());
    }
}


void GapSolver::presolve() {
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

template <typename PricerType, typename FarkasPricerType>
void GapSolver::solve(PricerType& pricer, FarkasPricerType& pricer_farkas) {
    total_time.start();
    tbl.write_header();

    presolve();

	if (_UB == _LB) {
		return;
	}

    bool should_stop = false;
    HandleCtrlC ctrl_c_handler([&]() { 
		if (should_stop) exit(-1);  // force stop

		std::cout << fmtquill::format("{} {}: Ctrl-C pressed, stopping...\n", instance.name, pricer.name);
        should_stop = true; 
    });

    rmp.reset(new Highs);
    rmp->setOptionValue("output_flag", false);
    rmp->setOptionValue(kPresolveString, "off");
    std::function<HighsInt()> get_lp_iters = [&]() { return rmp->getInfo().simplex_iteration_count; };

    auto model = SetCoverRestrictedProblem(instance.jobs, instance.machines, ObjSense::kMinimize);
    rmp->passModel(model);
    pricer.init(rmp.get(), &instance);
    column_management.init(rmp.get(), &instance);

    // initialize pricing
    pricing.init(instance);
    pricer_farkas.init(pricer, pricing, lp);

    double optimal_pricing = 0.0;
    bool any = false;

cg_time.start();
    pricer_farkas.optimize(lp._solution.row_dual, pricing, _reduced_costs);
    add_columns(rmp.get(), instance, pricing, _reduced_costs, _ones);
cg_time.pause();

    restoreFeasibility(pricer_farkas);
    pricer.init_feasible();

    double _rmpLB = rmp->getObjectiveValue();
    updateCompactSolution();
    tbl.output(0, 0, iteration_count, _LB, _UB, "-", _rmpLB, "-", basis_size, rmp->getNumCol(), total_time.TotalSeconds(), lp_iteration_count, fractional_count, prune_count, leaf_count);
    csv_writer.append_row(instance.name, pricer.name, 0, 0, iteration_count, _LB, _UB, "", _rmpLB, "", basis_size, rmp->getNumCol(), rmp_time.TotalSeconds(), cg_time.TotalSeconds(), total_time.TotalSeconds(), lp_iteration_count, "", -1);

    // primal simplex for warm-start "add columns"
    rmp->setOptionValue("simplex_strategy", "4");
    rmp->setOptionValue("allow_unbounded_or_infeasible", false);  // not sure if this adds unnecessary overheads

    int new_columns_index = rmp->getNumCol();
    size_t node_count = 0;

    if (params.nodes > 0) {
        do {
rmp_time.start();
            auto status = rmp->run();

            if (rmp->getModelStatus() != HighsModelStatus::kOptimal) {
                std::cout << fmtquill::format("{} {}: Error - {}\n", instance.name, pricer.name, (int)rmp->getModelStatus());
                return;
            }

            _rmpLB = rmp->getObjectiveValue();
            auto& solution = rmp->getSolution();

            updateCompactSolution();
            lp_iteration_count += get_lp_iters();
rmp_time.pause();

cg_time.start();
            pricer.update();
            optimal_pricing = pricer.optimize(solution.row_dual, pricing, _reduced_costs);

            column_management.reduce(iteration_count);

            new_columns_index = rmp->getNumCol();
            any = add_columns(rmp.get(), instance, pricing, _reduced_costs, _ones);

            // ASSUMES lambda <= 1, otherwise need to scale (e.g. _rmpLB / (1 + reduced cost))
            // This is only valid in root node, otherwise need to keep track of worst node
            _LB = std::max(_LB, _rmpLB - optimal_pricing);
cg_time.pause();

           // check if we can stop
            double lb = std::ceil(_LB - 1e-6);
            double gap = (_UB - lb) / _UB;

            if (gap < params.gap || lb + 1e-6 >= _rmpLB)
                break;

            // logging
            csv_writer.append_row(instance.name, pricer.name, 0, 0, iteration_count, _LB, _UB, gap * 100, _rmpLB, optimal_pricing, basis_size, rmp->getNumCol(), rmp_time.TotalSeconds(), cg_time.TotalSeconds(), total_time.TotalSeconds(), lp_iteration_count, "", 0);

            if (iteration_count % ITERATION_OUTPUT == 0 && total_time.TotalSeconds() - previous_logging_time > ITERATION_TIME) {
                tbl.output(0, 0, iteration_count, _LB, _UB, gap * 100, _rmpLB, optimal_pricing, basis_size, rmp->getNumCol(), total_time.TotalSeconds(), lp_iteration_count, fractional_count, prune_count, leaf_count);
                previous_logging_time = total_time.TotalSeconds();
            }

            ++iteration_count;

        } while (any && (params.timeout < 0 || total_time.TotalSeconds() < params.timeout) && !should_stop);

        updateCompactSolution();
    }

	double lb = std::ceil(_LB - 1e-6);
    double gap = std::abs(100.0 * (_UB - lb) / _UB);
    tbl.output(0, 0, iteration_count, lb, _UB, gap, _rmpLB, optimal_pricing, basis_size, rmp->getNumCol(), total_time.TotalSeconds(), lp_iteration_count, fractional_count, prune_count, leaf_count);
    csv_writer.append_row(instance.name, pricer.name, 0, 0, iteration_count, _LB, _UB, gap, _rmpLB, optimal_pricing, basis_size, rmp->getNumCol(), rmp_time.TotalSeconds(), cg_time.TotalSeconds(), total_time.TotalSeconds(), lp_iteration_count, "", 1);

	bool isOptimal = gap < params.gap;
	bool isTimeOut = params.timeout > 0 && total_time.TotalSeconds() > params.timeout;

    // branch if needed
    if (!isOptimal && !isTimeOut) {
        PricerType pricer_branch = pricer;

        openNodes.push_back({ _LB, {} });

        OpenNode node;

        do {
            ++node_count;

            if (node_count >= params.nodes)
                break;

            // update _LB
            double min_lower_bound = std::numeric_limits<double>::max();
            for (const auto& n : openNodes) {
                min_lower_bound = std::min(min_lower_bound, n.lower_bound);
            }

            _LB = std::max(_LB, min_lower_bound);

            // Lower bound
            std::vector<OpenNode>::iterator itNode = std::min_element(openNodes.begin(), openNodes.end(),
                [&](const OpenNode& a, const OpenNode& b) {
                    std_counter ac;
                    std::set_intersection(a.fixed_lb.begin(), a.fixed_lb.end(), node.fixed_lb.begin(), node.fixed_lb.end(), 
                        std::back_inserter(ac));

                    std_counter bc;
                    std::set_intersection(b.fixed_lb.begin(), b.fixed_lb.end(), node.fixed_lb.begin(), node.fixed_lb.end(),
                        std::back_inserter(bc));
                    
                    return a.lower_bound < b.lower_bound || a.lower_bound < b.lower_bound + 1e-6 && ac.count > bc.count;
                });

            node = *itNode;
            openNodes.erase(itNode);

            //
            // apply branching constraints
            //
            // 
            // update pricing
            for (int m = 0; m < instance.machines; ++m) {
                auto& lb = pricing[m].lb_bound;
                std::fill(lb.begin(), lb.end(), false);
            }

            for (auto& idx : node.fixed_lb) {
                pricing[idx / instance.jobs].lb_bound[idx % instance.jobs] = true;
            }

            // clear invalid columns
            std::vector<int> invalid_columns;

            for (size_t idx = 0, size = rmp->getNumCol(); idx < size; ++idx) {
                auto start = col_begin(*rmp, idx);
                auto end = --col_end(*rmp, idx);
                auto machine = *end - instance.jobs;

                // check if column is valid, i.e., if it satisfies the branching constraints
                bool valid = true;

                // fixed_lb must not be in the column
                for (auto& fixed : node.fixed_lb) {
                    if (machine == fixed / instance.jobs) {
                        int job = fixed % instance.jobs;

                        auto it = std::lower_bound(start, end, job);
                        bool found = it != end && job == *it;

                        if (found == true) {
                            valid = false;
                            break;
                        }
                    }
                }

                if (valid == false) {
                    invalid_columns.push_back(idx);
                }
            }


            for (int c = 0; c < rmp->getNumCol(); ++c) {
                rmp->changeColBounds(c, 0, kHighsInf);
            }

            for (int c : invalid_columns) {
                rmp->changeColBounds(c, 0, 0);
            }

            // restore feasibility or fathom infeasible nodes
            if (!restoreFeasibility(pricer_farkas)) {
                continue;
            }

            // solve CG / LP
            // update LB/UB if appropriate
            pricer_branch.init_feasible();

            double _nodeLB = rmp->getObjectiveValue();
            updateCompactSolution();

            // primal simplex for warm-start "add columns"
            rmp->setOptionValue("simplex_strategy", "4");
            rmp->setOptionValue("allow_unbounded_or_infeasible", false);  // not sure if this adds unnecessary overheads

            do {
                rmp_time.start();
                auto status = rmp->run();

                if (rmp->getModelStatus() != HighsModelStatus::kOptimal) {
                    std::cout << fmtquill::format("{} {}: Error - {}\n", instance.name, pricer.name, (int)rmp->getModelStatus());
                    return;
                }

                _nodeLB = rmp->getObjectiveValue();
                auto& solution = rmp->getSolution();
                updateCompactSolution();
                lp_iteration_count += get_lp_iters();
                rmp_time.pause();

                cg_time.start();
                pricer_branch.update();
                optimal_pricing = pricer_branch.optimize(solution.row_dual, pricing, _reduced_costs);

                column_management.reduce(iteration_count);
                any = add_columns(rmp.get(), instance, pricing, _reduced_costs, _ones);
                cg_time.pause();

                gap = (_UB - std::ceil(_LB - 1e-6)) / _UB;

                if (gap < params.gap || _LB + 1e-6 >= _nodeLB)
                    break;

                // logging
                csv_writer.append_row(instance.name, pricer_branch.name, node_count, openNodes.size(), iteration_count, _LB, _UB, gap * 100, _nodeLB, optimal_pricing, basis_size, rmp->getNumCol(), rmp_time.TotalSeconds(), cg_time.TotalSeconds(), total_time.TotalSeconds(), lp_iteration_count, "", 2);

                if (iteration_count % ITERATION_OUTPUT == 0 && total_time.TotalSeconds() - previous_logging_time > ITERATION_TIME) {
                    tbl.output(node_count, openNodes.size(), iteration_count, _LB, _UB, gap * 100, _nodeLB, optimal_pricing, basis_size, rmp->getNumCol(), total_time.TotalSeconds(), lp_iteration_count, fractional_count, prune_count, leaf_count);
                    previous_logging_time = total_time.TotalSeconds();
                }

                ++iteration_count;

            } while (any && (params.timeout < 0 || total_time.TotalSeconds() < params.timeout));

            //pricer_farkas._template.update(rmp->getSolution(), *rmp);

            updateCompactSolution();
            double gap = std::abs(100.0 * (_UB - std::ceil(_LB - 1e-6)) / _UB);
            csv_writer.append_row(instance.name, pricer_branch.name, node_count, openNodes.size(), iteration_count, _LB, _UB, gap, _rmpLB, optimal_pricing, basis_size, rmp->getNumCol(), rmp_time.TotalSeconds(), cg_time.TotalSeconds(), total_time.TotalSeconds(), lp_iteration_count, "", 3);

            if (iteration_count % ITERATION_OUTPUT == 0 && total_time.TotalSeconds() - previous_logging_time > ITERATION_TIME) {
                tbl.output(node_count, openNodes.size(), iteration_count, _LB, _UB, gap, _rmpLB, optimal_pricing, basis_size, rmp->getNumCol(), total_time.TotalSeconds(), lp_iteration_count, fractional_count, prune_count, leaf_count);
                previous_logging_time = total_time.TotalSeconds();
            }

            isOptimal = gap < params.gap;
            isTimeOut = params.timeout > 0 && total_time.TotalSeconds() > params.timeout;

            prune_count += _nodeLB >= _UB;
            leaf_count += isIntegral;

            if (isOptimal || isTimeOut) {
                _LB = std::ceil(_LB - 1e-6);
                tbl.output(node_count, openNodes.size(), iteration_count, _LB, _UB, gap, _rmpLB, optimal_pricing, basis_size, rmp->getNumCol(), total_time.TotalSeconds(), lp_iteration_count, fractional_count, prune_count, leaf_count);
                break;
            }

            // prune if node lb >= global UB
            if (_nodeLB >= _UB || isIntegral) {
                continue;
            }

            // choose new branching variable
            // TODO: improve choice!!

            // choose job most fractional machines
            int best_job = -1;
            double best_tie_breaker = 0;
            std::vector<int> branching_machines;
            std::vector<int> branching_tmp;

            for (int j = 0; j < instance.jobs; ++j) {
                double tie_break = 0;
                branching_tmp.clear();
                bool has_fractional = false;

                for (int m = 0; m < instance.machines; ++m) {
                    int idx = m * instance.jobs + j;

                    if (!HighsIntegers::isIntegral(_compact_solution[idx], 1e-6)) {
                        has_fractional = true;
                        tie_break += instance.profit[m][j] * _compact_solution[idx];
                        branching_tmp.push_back(idx);
                    }
                }

                if (has_fractional && (best_job == -1 || (branching_tmp.size() > branching_machines.size() || branching_tmp.size() == branching_machines.size() && tie_break > best_tie_breaker))) {
                //if (has_fractional && (best_job == -1 || tie_break < best_tie_breaker)) {
                    branching_machines.swap(branching_tmp);
                    best_tie_breaker = tie_break;
                    best_job = j;
                }
            }

            node.lower_bound = _nodeLB;

            // GUB branching, branch zeros only, but in subsets
            std::vector<HighsInt> remaining_machines;
            std::vector<HighsInt> remaining_machines_tmp;

            remaining_machines.reserve(instance.machines);

            for (int idx = best_job, size = instance.machines * instance.jobs; idx < size; idx += instance.jobs) {
				remaining_machines.push_back(idx);
            }

			std::set_difference(remaining_machines.begin(), remaining_machines.end(), node.fixed_lb.begin(), node.fixed_lb.end(),
                std::back_inserter(remaining_machines_tmp));

			remaining_machines.swap(remaining_machines_tmp);
			remaining_machines_tmp.clear();

            std::set_difference(remaining_machines.begin(), remaining_machines.end(), branching_machines.begin(), branching_machines.end(),
                std::back_inserter(remaining_machines_tmp));

            remaining_machines.swap(remaining_machines_tmp);

			// need to partition these machines into two sets, should be a superset of branching_machines
			// need to partition branching_machines into these two sets, try to balance its mass
            double d1 = 0, d2 = 0;
			int i1 = 0, i2 = 0;
            OpenNode p1 = node;
            OpenNode p2 = node;

            if (branching_machines.size() == 2) {
                p1.fixed_lb.push_back(branching_machines[0]);
                p2.fixed_lb.push_back(branching_machines[1]);
            }
            else {
                // sort in descending order for better balanced partition
                std::stable_sort(branching_machines.begin(), branching_machines.end(), [&](int a, int b) {
                    return _compact_solution[a] > _compact_solution[b];
                });

                // greedy balance
                for (int idx : branching_machines) {
                    if (d1 < d2) {
                        p1.fixed_lb.push_back(idx);
                        d1 += _compact_solution[idx];
                        ++i1;
                    }
                    else {
                        p2.fixed_lb.push_back(idx);
                        d2 += _compact_solution[idx];
                        ++i2;
                    }
                }
            }

			// assign the remaining machines (greedily)
            d1 = d2 = 0;
            // sort in descending order for better balanced partition
            std::stable_sort(remaining_machines.begin(), remaining_machines.end(), [&](int a, int b) {
                int ma = a / instance.jobs;
                int mb = b / instance.jobs;

                double sa = instance.demands[ma][best_job] / instance.capacity[ma];
                double sb = instance.demands[mb][best_job] / instance.capacity[mb];

                return instance.profit[ma][best_job] * sa > instance.profit[mb][best_job] * sb;
            });

			for (int idx : remaining_machines) {
				if (d1 < d2) {
					p1.fixed_lb.push_back(idx);
					++i1;
                    d1 += instance.profit[idx / instance.jobs][best_job];
				}
				else {
					p2.fixed_lb.push_back(idx);
					++i2;
                    d2 += instance.profit[idx / instance.jobs][best_job];
                }
			}

			std::stable_sort(p1.fixed_lb.begin(), p1.fixed_lb.end());
            std::stable_sort(p2.fixed_lb.begin(), p2.fixed_lb.end());

            openNodes.push_back(p1);
            openNodes.push_back(p2);

        } while (openNodes.empty() == false);
    }

    csv_writer.append_row(instance.name, pricer.name, node_count, openNodes.size(), iteration_count, _LB, _UB, gap, _rmpLB, optimal_pricing, basis_size, rmp->getNumCol(), rmp_time.TotalSeconds(), cg_time.TotalSeconds(), total_time.TotalSeconds(), lp_iteration_count, "", 4);

    std::cout << fmtquill::format("\n"
		"Inst : {}\n"
		"Price: {}\n"
        "RMP  : {:.3f} s\n" 
        "CG   : {:.3f} s\n" 
		"Total: {:.3f} s\n" 
		"#Cols: {}\n"  
        "Gap  : {:.2f}% \n", instance.name, pricer.name, rmp_time.TotalSeconds(), cg_time.TotalSeconds(), total_time.TotalSeconds(), rmp->getNumCol(), gap);
}

template <typename FarkasPricerType>
bool GapSolver::restoreFeasibility(FarkasPricerType &pricer_farkas) {
    bool has_dual_ray = false;
    std::vector<double> dual_ray(rmp->getNumRow(), 1);

    rmp->setOptionValue("simplex_strategy", "1"); // need to use dual solver for extreme ray
    rmp->setOptionValue("allow_unbounded_or_infeasible", true);

    // tight loop to restore feasibility (assuming possible!)
    do {
rmp_time.start();
        //rmp->clearSolver();
        auto status = rmp->run();

        // HiGHs has gotten into a bad state, so we need to reset?
        if (status == HighsStatus::kError) {
            std::cout << fmtquill::format("{}: Unrecoverable Error\n", instance.name);
            return false;
        }

        rmp->getDualRay(has_dual_ray, dual_ray.data());
        lp_iteration_count += rmp->getInfo().simplex_iteration_count;

        if (rmp->getModelStatus() == HighsModelStatus::kUnknown && has_dual_ray == false) {
            std::cout << fmtquill::format("{}: Test Error - {}\n", instance.name, (int)rmp->getModelStatus());
            return false;
        }
rmp_time.pause();

cg_time.start();
        if (has_dual_ray == true || rmp->getNumCol() == 0) {
            if (pricer_farkas.optimize(dual_ray, pricing, _reduced_costs) > -kHighsInf)
                add_columns(rmp.get(), instance, pricing, _reduced_costs, _ones);
            else {
				std::cout << "Node infeasible!" << std::endl;
                return false;
            }
        }
cg_time.pause();

        // debugging
        if (iteration_count % ITERATION_OUTPUT == 0 && total_time.TotalSeconds() - previous_logging_time > ITERATION_TIME && has_dual_ray) {
            tbl.output("", "", iteration_count, _LB, "-", "-", "-", "-", "-", rmp->getNumCol(), total_time.TotalSeconds(), lp_iteration_count, fractional_count, prune_count, leaf_count);
            previous_logging_time = total_time.TotalSeconds();
        }

        ++iteration_count;
    } while (has_dual_ray == true);

    return true;
}

void GapSolver::updateCompactSolution() {
    const auto& solution = rmp->getSolution();

    basis_size = 0;
	std::fill(_compact_solution.begin(), _compact_solution.end(), 0);

    // try to find UB solution
    for (size_t idx = 0, size = rmp->getNumCol(); idx < size; ++idx) {
        if (solution.col_value[idx] > 1e-6) {
            ++basis_size;
            auto end = --col_end(*rmp, idx);
            auto machine = *end - instance.jobs;

            for (auto it = col_begin(*rmp, idx); it != end; ++it) {
                _compact_solution[machine * instance.jobs + *it] += solution.col_value[idx];
            }
        }
    }

	// check if integral
	//isIntegral = true;
    fractional_count = 0;

	for (double val : _compact_solution) {
        fractional_count += size_t(val > 1e-6 && val < 1 - 1e-6);
	}

    isIntegral = fractional_count == 0;

	if (isIntegral) {
		double tmpUB = 0;
		for (int m = 0; m < instance.machines; ++m) {
			auto& profit = instance.profit[m];
			for (int j = 0; j < instance.jobs; ++j) {
				tmpUB += _compact_solution[m * instance.jobs + j] > 1e-6 ? profit[j] : 0;
			}
		}

		if (_UB > tmpUB) {
			_UB = tmpUB;
		}
	}

    //bool hasDuplicates = false;
    //for (int j = 0; j < instance.jobs; ++j) {
    //    double tmp = 0;
    //    for (int m = 0; m < instance.machines; ++m) {
    //        tmp += _compact_solution[m * instance.jobs + j];
    //    }

    //    if (tmp > 1 + 1e-6) {
    //        hasDuplicates = true;
    //        break;
    //    }
    //}

    if (basis_size == instance.machines) {
        double tmpUB = remove_duplicates(instance, rmp.get());

        if (_UB > tmpUB) {
			_UB = tmpUB;
            //_compact_solution_best = _compact_solution;
        }

        //isIntegral = true;
    }
}

double remove_duplicates(GapInstance& instance, Highs* rmp) {
    double cost = 0;
	auto& solution = rmp->getSolution();
    std::vector<std::vector<int>> job_machine(instance.jobs);

    // check duplicates
	// we assume that the last element of a column is the machine index
    for (size_t idx = 0, size = rmp->getNumCol(); idx < size; ++idx) {
        if (solution.col_value[idx] > 1e-6) {
            auto end = --col_end(*rmp, idx);
            auto machine = *end - instance.jobs;

            for (auto it = col_begin(*rmp, idx); it != end; ++it) {
                job_machine[*it].push_back(machine);
                cost += instance.profit[machine][*it];
            }
        }
    }

    // remove duplicates, keep cheapest one
    std::vector<std::vector<int>> partition(instance.machines);
	bool any = false;

    for (int j = 0; j < instance.jobs; ++j) {
        if (job_machine[j].size() > 1) {
            double min_cost = std::numeric_limits<double>::max();
            double total_cost = 0;
			int best_machine = -1;

            for (int m : job_machine[j]) {
                total_cost += instance.profit[m][j];

                if (min_cost > instance.profit[m][j]) {
                    min_cost = instance.profit[m][j];
                    best_machine = m;
                }
            }

            cost -= (total_cost - min_cost);
            partition[best_machine].push_back(j);
			any = true;
        }
		else if (job_machine[j].size() == 1)
			partition[job_machine[j][0]].push_back(j);
    }

    if (any) {
        // add modified columns back into RMP to get partition instead of cover
        std::vector<double> ones(instance.jobs + 1, 1);

        for (int m = 0; m < instance.machines; ++m) {
            auto& profit = instance.profit[m];
            double cost = 0;

            for (auto j : partition[m]) {
                cost += profit[j];
            }

            partition[m].push_back(instance.jobs + m);
            rmp->addCol(cost, 0, kHighsInf, partition[m].size(), partition[m].data(), ones.data());
        }

        rmp->run();
    }

    return cost;
}

bool add_columns(Highs* rmp, GapInstance& instance, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs, std::vector<double>& ones) {
    bool any = false;

    for (int m = 0; m < instance.machines; ++m) {
        auto& profit = instance.profit[m];

        if (reduced_costs[m] > 1e-6) {
            std::stable_sort(pricing[m].solution.begin(), pricing[m].solution.end()); // ensure columns are sorted for faster search

            double cost = 0;

			for (int i = 0, size = pricing[m].solution.size() - 1; i < size; ++i) {
				cost += profit[pricing[m].solution[i]];
			}

            rmp->addCol(cost, 0, kHighsInf, pricing[m].solution.size(), pricing[m].solution.data(), ones.data());
            any = true;
        }
    }

    return any;
}

