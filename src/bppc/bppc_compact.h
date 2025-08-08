#pragma once
#include "bppc_instance.h"
#include "highs/Highs.h"

struct BppcCompact {
	BppcInstance& _instance;
	HighsSolution _solution;
	size_t iterations = 0;
	std::unique_ptr<Highs> highs;
	std::vector<std::vector<double>> lb_template_pricing;

	BppcCompact(BppcInstance& instance) : _instance(instance) {
		highs = std::make_unique<Highs>();
	}

	// greedy first-fit algorithm
	std::vector<std::vector<int>> heuristic() {
		std::vector<int> items;

		items.resize(_instance.items);
		std::iota(items.begin(), items.end(), 0);

		//items = GraphColorPricingBdd::max_connected_degree(instance.start, instance.index);
		//std::stable_sort(items.begin(), items.end(), [&instance](int i, int j) { return instance.item_weights[i] > instance.item_weights[j]; });

		std::vector<std::tuple<int, std::vector<int>>> bins;

		// greedy first-fit
		for (auto index : items) {
			bool packed = false;
			for (auto& [size, items] : bins) {

				// check if we can pack the item
				if (size + _instance.weights[index] <= _instance.capacity) {

					// check conflict graph.  Can assume graph items are sorted by index, so can use binary search
					bool conflict = false;

					for (auto i : items) {
						// is i in the conflict graph of index?
						if (_instance.has_conflict(index, i)) {
							conflict = true;
							break;
						}
					}

					if (!conflict) {
						size += _instance.weights[index];
						items.push_back(index);
						packed = true;
						break;
					}
				}
			}

			if (!packed) {
				bins.push_back({ _instance.weights[index], { index } });
			}
		}

		std::vector<std::vector<int>> result;
		result.reserve(bins.size());

		for (auto& [size, items] : bins) {
			std::sort(items.begin(), items.end());
			result.emplace_back(std::move(items));
		}

		return result;
	}

	double solve(int BINS = -1) {
		auto initial_assignment = heuristic();

		if (BINS == -1) {
			BINS = initial_assignment.size();
		}

		// try to model the problem as a compact MIP
		// min sum y_b
		// s.t. 
		//                     sum_b { x_ib } == 1, i = 1, ..., ITEMS
		// -inf <= sum_i { w_i x_ib } - C y_b <= 0, b = 1, ..., BINS
		//    0 <= x_ib + x_jb                <= 1, (i,j) in CONFLICTS, b = 1, ..., BINS
		// 
		// x_ib in {0, 1}, b = 1, ..., BINS, i = 1, ..., ITEMS
		// y_b in {0, 1}, b = 1, ..., BINS

		HighsModel compact_model;

		int ITEMS = _instance.items;
		int CONFLICTS = std::accumulate(_instance.conflict_graph.begin(), _instance.conflict_graph.end(), 0, 
			[](int sum, const auto& vec) { return sum + (int)vec.size(); });

		compact_model.lp_.num_col_ = BINS + BINS * ITEMS;
		compact_model.lp_.num_row_ = ITEMS + BINS + CONFLICTS * BINS;
		compact_model.lp_.offset_ = 0;
		compact_model.lp_.sense_ = ObjSense::kMinimize;

		// columns
		compact_model.lp_.col_lower_.resize(compact_model.lp_.num_col_, 0);
		compact_model.lp_.col_upper_.resize(compact_model.lp_.num_col_, 1);

		// rows
		compact_model.lp_.row_lower_.resize(compact_model.lp_.num_row_, 0);
		compact_model.lp_.row_upper_.resize(compact_model.lp_.num_row_, 1);

		// objective
		compact_model.lp_.col_cost_.assign(compact_model.lp_.num_col_, 0);
		std::fill(compact_model.lp_.col_cost_.begin(), compact_model.lp_.col_cost_.begin() + BINS, 1); // y_b

		// initialize empty matrix
		compact_model.lp_.a_matrix_.num_col_ = compact_model.lp_.num_col_;
		compact_model.lp_.a_matrix_.num_row_ = compact_model.lp_.num_row_;
		compact_model.lp_.a_matrix_.format_ = MatrixFormat::kRowwise;
		compact_model.lp_.a_matrix_.start_ = { 0 };

		auto Y_b = [&](int b) { return b; };
		auto X_ib = [&](int i, int b) { return BINS + b * ITEMS + i; };

		int row = 0;

		// assignment constraints
		for (int i = 0; i < ITEMS; ++i) {
			for (int b = 0; b < BINS; ++b) {
				compact_model.lp_.a_matrix_.index_.push_back(X_ib(i, b));
				compact_model.lp_.a_matrix_.value_.push_back(1);
			}

			compact_model.lp_.a_matrix_.start_.push_back(compact_model.lp_.a_matrix_.index_.size());
			compact_model.lp_.row_lower_[row] = 1; // assignment
			++row;
		}

		// capacity constraints
		for (int b = 0; b < BINS; ++b) {
			for (int i = 0; i < ITEMS; ++i) {
				compact_model.lp_.a_matrix_.index_.push_back(X_ib(i, b));
				compact_model.lp_.a_matrix_.value_.push_back(_instance.weights[i]);
			}

			compact_model.lp_.a_matrix_.index_.push_back(Y_b(b));
			compact_model.lp_.a_matrix_.value_.push_back(-_instance.capacity);

			compact_model.lp_.a_matrix_.start_.push_back(compact_model.lp_.a_matrix_.index_.size());

			compact_model.lp_.row_lower_[row] = -kHighsInf;
			compact_model.lp_.row_upper_[row] = 0;
			++row;
		}

		// conflict constraints
		for (int b = 0; b < BINS; ++b) {
			for (int i = 0; i < ITEMS; ++i) {
				for (auto jit = _instance.conflict_graph[i].begin(); jit != _instance.conflict_graph[i].end(); ++jit) {
					int j = *jit;
					compact_model.lp_.a_matrix_.index_.push_back(X_ib(i, b));
					compact_model.lp_.a_matrix_.value_.push_back(1);
					compact_model.lp_.a_matrix_.index_.push_back(X_ib(j, b));
					compact_model.lp_.a_matrix_.value_.push_back(1);
					compact_model.lp_.a_matrix_.start_.push_back(compact_model.lp_.a_matrix_.index_.size());
				}
			}
		}

		//compact_model.lp_.integrality_.assign(compact_model.lp_.num_col_, HighsVarType::kInteger);

		highs->passModel(compact_model);

		HighsSolution insol;
		insol.col_value.resize(compact_model.lp_.num_col_, 0.0);

		for (int b = 0; b < BINS; ++b) {
			insol.col_value[Y_b(b)] = 1.0; // use bin

			for (int i : initial_assignment[b]) {
				insol.col_value[X_ib(i, b)] = 1.0; // assign items to bins
			}
		}

		//highs->setSolution(insol);
		highs->setOptionValue("time_limit", 20);

		//highs->writeModel("bppc.mps");
		highs->run();

		std::cout << "LP Objective: " << highs->getObjectiveValue() << std::endl;

		auto& sol = highs->getSolution();

		std::vector<std::vector<int>> lb_solution(BINS);
		lb_template_pricing.resize(BINS, std::vector<double>(_instance.items, -1.0));

		// print out solution
		for (int b = 0; b < BINS; ++b) {
			std::cout << "Bin " << b << ": " << std::endl;
			for (int i = 0; i < ITEMS; ++i) {
				auto value = sol.col_value[X_ib(i, b)];

				if (value > 1e-6) {
					std::cout << i << " " << sol.col_value[X_ib(i, b)] << std::endl;

					if (value > 1.0 - 1e-6) {
						lb_solution[b].push_back(i);
						lb_template_pricing[b][i] = 1.0;
					}
					else {
						lb_template_pricing[b][i] = 0;
					}
				}
			}
			std::cout << std::endl;
		}

		return highs->getObjectiveValue();

		//auto& sol = solver.getSolution();

		//std::vector<std::vector<int>> lb_solution(BINS);
		//std::vector<std::vector<double>> lb_template_pricing(BINS, std::vector<double>(instance.item_weights.size(), -1.0));

		//// print out solution
		//for (int b = 0; b < BINS; ++b) {
		//	std::cout << "Bin " << b << ": " << std::endl;
		//	for (int i = 0; i < ITEMS; ++i) {
		//		auto value = sol.col_value[X_ib(i, b)];

		//		if (value > 1e-6) {
		//			std::cout << i << " " << sol.col_value[X_ib(i, b)] << std::endl;

		//			if (value > 1.0 - 1e-6) {
		//				lb_solution[b].push_back(i);
		//				lb_template_pricing[b][i] = 1.0;
		//			}
		//			else {
		//				lb_template_pricing[b][i] = 0;
		//			}
		//		}
		//	}
		//	std::cout << std::endl;
		//}
	}
};
