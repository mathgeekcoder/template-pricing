#include <vector>
#include <algorithm>
#include "highs/util/HighsIntegers.h"
#include <numeric>
#include <ctime>
#include "bppc.h"
#include "bppc_instance.h"
#include <filesystem>
#include "quill/bundled/fmt/format.h"
#include "utils.h"

// Supported template instantiations
template int BppcSolver::solve(BppcTemplatePrice& pricer);
template int BppcSolver::solve(BppcDantzigPrice& pricer);

void BppcSolver::presolve() {
	// try to fit entire item into bins, then split items to achieve lower bound
	std::vector<int> items(instance.items);
	std::iota(items.begin(), items.end(), 0);
	std::stable_sort(items.begin(), items.end(), [&](int i, int j) { return instance.weights[i] > instance.weights[j]; });

	std::vector<std::tuple<int, std::vector<int>>> bins;

	// greedy first-fit
	for (auto index : items) {
		bool packed = false;
		for (auto& [size, items] : bins) {
			// check if we can pack the item
			if (size + instance.weights[index] <= instance.capacity) {
				bool conflict = false;

				for (auto i : items) {
					if (instance.has_conflict(index, i)) {
						conflict = true;
						break;
					}
				}

				if (!conflict) {
					size += instance.weights[index];
					items.push_back(index);
					packed = true;
					break;
				}
			}
		}

		if (!packed) {
			bins.push_back({ instance.weights[index], { index } });
		}
	}

	// sort bins by size
	std::sort(bins.begin(), bins.end(), [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });

	std::vector<std::vector<double>> template_pricing(bins.size(), std::vector<double>(instance.items, -1.0));

	for (int i = bins.size() - 1; i >= 0; --i) {
		for (auto j : std::get<1>(bins[i])) {
			template_pricing[i][j] = 1.0;
		}
	}

	// try to split last bins to previous bins
	while (true) {
		// try to add, abort if not possible
		auto tmp_bins = bins;  // copy bins
		auto [last_size, last_items] = tmp_bins.back();
		tmp_bins.pop_back();

		for (auto index : last_items) {
			double remaining_item = instance.weights[index];

			// split last bin to previous bins
			for (int i = 0; i < tmp_bins.size() && remaining_item > 0; ++i) {
				auto& [size, items] = tmp_bins[i];

				if (size < instance.capacity) {
					double remaining_bin = instance.capacity - size;
					double added = std::min(remaining_bin, remaining_item);

					template_pricing[i][index] = 0.0;
					remaining_item -= added;
					size += added;
					last_size -= added;

					//items.push_back(index); // don't add fractional items to initial columns
				}
			}
		}

		if (last_size == 0) {
			bins = std::move(tmp_bins);
		}
		else {
			break;
		}
	}

	std::vector<std::vector<int>> result;
	result.reserve(bins.size());

	for (auto& [size, items] : bins) {
		std::sort(items.begin(), items.end());
		result.emplace_back(std::move(items));
	}

	template_pricing.resize(bins.size());
	//return { result, template_pricing };


	// calculate rough lower bound
	double lower_bound = 0;
	for (auto w : instance.weights) {
		lower_bound += w;
	}
	lower_bound /= instance.capacity;

	// test lower bound on clique (instead of bin packing)
	uint32_t num_vertices = instance.weights.size();

	int max_degree = 0;
	int max_degree_vertex = 0;

	//for (int col = 0; col < num_vertices; ++col) {
	//	if (max_degree < instance.start[col + 1] - instance.start[col]) {
	//		max_degree = instance.start[col + 1] - instance.start[col];
	//		max_degree_vertex = col;
	//	}
	//}

	int clique_lower_bound = 1;
	//std::vector<int> remaining(instance.index.begin() + instance.start[max_degree_vertex], instance.index.begin() + instance.start[max_degree_vertex + 1]);

	//// choose next variable from remaining items that is most connected to other remaining items. repeat
	//while (remaining.empty() == false) {
	//	int remaining_max = -1;
	//	int remaining_max_vertex = -1;

	//	for (int i = 0, size = remaining.size(); i < size; ++i) {
	//		auto begin = instance.index.begin() + instance.start[remaining[i]];
	//		auto end = instance.index.begin() + instance.start[remaining[i] + 1];

	//		std_counter degree_remaining;
	//		std::set_intersection(begin, end, remaining.begin(), remaining.end(), std::back_inserter(degree_remaining));

	//		if (remaining_max < (int)degree_remaining.size()) {
	//			remaining_max = degree_remaining.size();
	//			remaining_max_vertex = remaining[i];
	//		}
	//	}

	//	// remove items not connected to remaining_max_vertex, i.e., keep only items that are connected to remaining_max_vertex
	//	std::vector<int> tmp;
	//	auto begin = instance.index.begin() + instance.start[remaining_max_vertex];
	//	auto end = instance.index.begin() + instance.start[remaining_max_vertex + 1];

	//	std::set_intersection(begin, end, remaining.begin(), remaining.end(), std::back_inserter(tmp));
	//	remaining = std::move(tmp);

	//	++clique_lower_bound;
	//}

	std::cout << "Capacity LB: " << (int)lower_bound << std::endl;
	std::cout << "Conflict LB: " << clique_lower_bound << std::endl;

	lower_bound = std::max(lower_bound, (double)clique_lower_bound);
}

template <typename PricerType>
int BppcSolver::solve(PricerType& pricer) {
	total_time.start();
	tbl.write_header();

	BppcCompact compact(instance);
	auto initial = compact.heuristic();
	int compactlb = std::ceil(compact.solve() - 1e-6);
	compact.solve(compactlb);
	//return 0;

	presolve();
	if (_UB == _LB) {
		return 0;
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
	rmp->setOptionValue("random_seed", params.random_seed);
	std::function<HighsInt()> get_lp_iters = [&]() { return rmp->getInfo().simplex_iteration_count; };

	auto model = SetCoverRestrictedProblem(instance.items, 0, ObjSense::kMinimize);
	rmp->passModel(model);
	pricer._machines = compactlb;
	pricer.init(rmp.get(), &instance);
//	column_management.init(rmp.get(), &instance);

	//for (auto& col : initial) {
	//	rmp->addCol(1, 0, kHighsInf, col.size(), col.data(), _ones.data());
	//}

	// initialize pricing
	pricing.init(instance);
//	pricer_farkas.init(pricer, pricing, lp);

	_reduced_costs.assign(compactlb, 0);

	bool has_dual_ray = false;
	std::vector<double> dual_ray(rmp->getNumRow(), 1);

	rmp->setOptionValue("simplex_strategy", "1"); // need to use dual solver for extreme ray
	rmp->setOptionValue("allow_unbounded_or_infeasible", true);

	pricer._template._template_columns = compact.lb_template_pricing;

	// initialize RMP if empty
	cg_time.start();
	if (rmp->getNumCol() == 0 && pricer.optimize(dual_ray, pricing, _reduced_costs) > -kHighsInf) {
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
			if (pricer.optimize(dual_ray, pricing, _reduced_costs) > -kHighsInf) {
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


	double optimal_pricing = 0.0;
	bool any = false;

	rmp->run();

	double _rmpLB = rmp->getObjectiveValue();
	updateCompactSolution();
	tbl.output(iteration_count, _LB, _UB, "-", _rmpLB, "-", basis_size, rmp->getNumCol(), total_time.TotalSeconds(), lp_iteration_count, fractional_count);
	csv_writer.append_row(instance.name, pricer.name, 0, 0, iteration_count, _LB, _UB, "", _rmpLB, "", basis_size, rmp->getNumCol(), rmp_time.TotalSeconds(), cg_time.TotalSeconds(), total_time.TotalSeconds(), lp_iteration_count, "", -1);

	// primal simplex for warm-start "add columns"
	rmp->setOptionValue("simplex_strategy", "4");
	rmp->setOptionValue("allow_unbounded_or_infeasible", false);  // not sure if this adds unnecessary overheads
	//rmp->setOptionValue(kSolverString, kIpmString);
	//rmp->setOptionValue(kRunCrossoverString, kHighsOffString);
	pricer.init_feasible();

	int new_columns_index = rmp->getNumCol();
	size_t node_count = 0;

	if (params.nodes > 0) {
		do {
			rmp_time.start();
			auto status = rmp->run();

			if (rmp->getModelStatus() != HighsModelStatus::kOptimal) {
				std::cout << fmtquill::format("{} {}: Error - {}\n", instance.name, pricer.name, (int)rmp->getModelStatus());
				return -1;
			}

			_rmpLB = rmp->getObjectiveValue();
			auto& solution = rmp->getSolution();

			updateCompactSolution();
			lp_iteration_count += get_lp_iters();
			rmp_time.pause();

			cg_time.start();
			pricer.update();
			optimal_pricing = pricer.optimize(solution.row_dual, pricing, _reduced_costs);

//			column_management.reduce(iteration_count);

			new_columns_index = rmp->getNumCol();
			any = add_columns(_reduced_costs);

			// need to scale (e.g. _rmpLB / (1 + reduced cost))
			// This is only valid in root node, otherwise need to keep track of worst node
			_LB = std::max(_LB, _rmpLB / (1 + optimal_pricing));
			cg_time.pause();

			// check if we can stop
			double lb = std::ceil(_LB - 1e-6);
			double gap = (_UB - lb) / _UB;

			if (gap < params.gap || lb + 1e-6 >= _rmpLB)
				break;

			// logging
			csv_writer.append_row(instance.name, pricer.name, 0, 0, iteration_count, _LB, _UB, gap * 100, _rmpLB, optimal_pricing, basis_size, rmp->getNumCol(), rmp_time.TotalSeconds(), cg_time.TotalSeconds(), total_time.TotalSeconds(), lp_iteration_count, "", 0);

			if (iteration_count % ITERATION_OUTPUT == 0 && total_time.TotalSeconds() - previous_logging_time > ITERATION_TIME) {
				tbl.output(iteration_count, _LB, _UB, gap * 100, _rmpLB, optimal_pricing, basis_size, rmp->getNumCol(), total_time.TotalSeconds(), lp_iteration_count, fractional_count);
				previous_logging_time = total_time.TotalSeconds();
			}

			++iteration_count;

		} while (any && (params.timeout < 0 || total_time.TotalSeconds() < params.timeout) && !should_stop);

		updateCompactSolution();
	}

	double lb = std::ceil(_LB - 1e-6);
	double gap = std::abs(100.0 * (_UB - lb) / _UB);
	tbl.output(iteration_count, lb, _UB, gap, _rmpLB, optimal_pricing, basis_size, rmp->getNumCol(), total_time.TotalSeconds(), lp_iteration_count, fractional_count);
	csv_writer.append_row(instance.name, pricer.name, 0, 0, iteration_count, _LB, _UB, gap, _rmpLB, optimal_pricing, basis_size, rmp->getNumCol(), rmp_time.TotalSeconds(), cg_time.TotalSeconds(), total_time.TotalSeconds(), lp_iteration_count, "", 1);

	bool isOptimal = gap < params.gap;
	csv_writer.append_row(instance.name, pricer.name, 0, 0, iteration_count, _LB, _UB, gap, _rmpLB, optimal_pricing, basis_size, rmp->getNumCol(), rmp_time.TotalSeconds(), cg_time.TotalSeconds(), total_time.TotalSeconds(), lp_iteration_count, "", 4);

	std::cout << fmtquill::format("\n"
		"Inst : {}\n"
		"Price: {}\n"
		"RMP  : {:.3f} s\n"
		"CG   : {:.3f} s\n"
		"Total: {:.3f} s\n"
		"#Cols: {}\n"
		"Gap  : {:.2f}% \n", instance.name, pricer.name, rmp_time.TotalSeconds(), cg_time.TotalSeconds(), total_time.TotalSeconds(), rmp->getNumCol(), gap);

	return 0;
}

void BppcSolver::updateCompactSolution() {
	const auto& solution = rmp->getSolution();

	if (solution.value_valid) {
		basis_size = 0;

		// try to find UB solution
		for (size_t idx = 0, size = rmp->getNumCol(); idx < size; ++idx) {
			if (solution.col_value[idx] > 1e-6) {
				++basis_size;
			}
		}

		if (basis_size == std::ceil(rmp->getObjectiveValue() - 1e-6)) {
			_UB = basis_size;
		}
	}
}

bool BppcSolver::add_columns(std::vector<double>& reduced_costs) {
	bool any = false;

	for (int m = 0; m < reduced_costs.size(); ++m) {
		if (reduced_costs[m] > 1e-6) {
			auto& solution = pricing[m].solution;
			auto test = rmp->addCol(1, 0, kHighsInf, static_cast<int>(solution.size()), solution.data(), _ones.data());
			any = true;
		}
	}

	return any;
}
