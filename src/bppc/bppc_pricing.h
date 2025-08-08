#pragma once
#include "bppc_instance.h"
#include "block/column_generation.h"
#include "highs/Highs.h"
#include "highs/parallel/HighsParallel.h"

// solve using MIP
struct BppcPricing {
    size_t _machine = 0;
    const BppcInstance* _instance = nullptr;
    std::vector<int> solution;
	std::unique_ptr<Highs> highs;

    void init(uint32_t index, const BppcInstance& instance) {
        _instance = &instance;
        _machine = index;
		highs = std::make_unique<Highs>();
		highs->setOptionValue("output_flag", false);
		highs->passModel(subproblem());
	}

    double optimize(const std::vector<double>& obj, double offset) {
		highs->changeColsCost(0, _instance->items - 1, obj.data());
		highs->run();

		solution.clear();
		auto& sol = highs->getSolution();
		
		for (int i = 0; i < _instance->items; ++i) {
			if (sol.col_value[i] > 1e-6) {
				this->solution.push_back(i);
			}
		}

		return highs->getObjectiveValue() + offset;
    }

    HighsModel subproblem() {
		HighsModel compact_model;

		int ITEMS = _instance->items;
		int CONFLICTS = std::accumulate(_instance->conflict_graph.begin(), _instance->conflict_graph.end(), 0, 
			[](int sum, const auto& vec) { return sum + (int)vec.size(); });

		// max sum_i { c_i x_i }
		// s.t. 
		// -inf <= sum_i { w_i x_ib } <= C
		//    0 <= x_ib + x_jb        <= 1, (i,j) in CONFLICTS
		// 
		// x_i in {0, 1}, i = 1, ..., ITEMS

		compact_model.lp_.num_col_ = ITEMS;
		compact_model.lp_.num_row_ = 1 + CONFLICTS;
		compact_model.lp_.offset_ = 0;
		compact_model.lp_.sense_ = ObjSense::kMaximize;

		// columns
		compact_model.lp_.col_lower_.resize(compact_model.lp_.num_col_, 0);
		compact_model.lp_.col_upper_.resize(compact_model.lp_.num_col_, 1);

		// rows
		compact_model.lp_.row_lower_.resize(compact_model.lp_.num_row_, 0);
		compact_model.lp_.row_upper_.resize(compact_model.lp_.num_row_, 1);

		compact_model.lp_.row_lower_[0] = -kHighsInf;
		compact_model.lp_.row_upper_[0] = _instance->capacity;

		// objective
		compact_model.lp_.col_cost_.assign(compact_model.lp_.num_col_, 0); // will be set to dual values

		// initialize empty matrix
		compact_model.lp_.a_matrix_.num_col_ = compact_model.lp_.num_col_;
		compact_model.lp_.a_matrix_.num_row_ = compact_model.lp_.num_row_;
		compact_model.lp_.a_matrix_.format_ = MatrixFormat::kRowwise;
		compact_model.lp_.a_matrix_.start_ = { 0 };

		// capacity constraints
		for (int i = 0; i < ITEMS; ++i) {
			compact_model.lp_.a_matrix_.index_.push_back(i);
			compact_model.lp_.a_matrix_.value_.push_back(_instance->weights[i]);
		}

		compact_model.lp_.a_matrix_.start_.push_back(compact_model.lp_.a_matrix_.index_.size());

		// conflict constraints
		for (int i = 0; i < ITEMS; ++i) {
			for (auto jit = _instance->conflict_graph[i].begin(); jit != _instance->conflict_graph[i].end(); ++jit) {
				int j = *jit;
				compact_model.lp_.a_matrix_.index_.push_back(i);
				compact_model.lp_.a_matrix_.value_.push_back(1);
				compact_model.lp_.a_matrix_.index_.push_back(j);
				compact_model.lp_.a_matrix_.value_.push_back(1);
				compact_model.lp_.a_matrix_.start_.push_back(compact_model.lp_.a_matrix_.index_.size());
			}
		}

		compact_model.lp_.integrality_.assign(compact_model.lp_.num_col_, HighsVarType::kInteger);
		return compact_model;
	}
};
