#pragma once
#include "gap_instance.h"
#include "highs/Highs.h"

template <typename RmpSolver>
struct GapCompact {
	const GapInstance& _instance;
	HighsSolution _solution;
	size_t iterations = 0;
	std::unique_ptr<RmpSolver> highs;

public:	
	GapCompact(const GapInstance& instance) : _instance(instance) {
		highs = std::make_unique<RmpSolver>();
	}

	double solve(bool output = false, bool integer = false) {
		highs->setOptionValue("output_flag", output);

		HighsModel model = create_model(integer);
		highs->passModel(model);
		highs->run();

		_solution = highs->getSolution();
		iterations = highs->getInfo().simplex_iteration_count;
		return highs->getObjectiveValue();
	}

	void terminate() {
		if constexpr (std::is_same<RmpSolver, Highs>::value) {
			//highs->terminate();
		}
		else {
			highs->requestStop();
		}
	}

private:	
	HighsModel create_model(bool integer) {
		HighsModel model;
		size_t M = _instance.machines;
		size_t J = _instance.jobs;

		model.lp_.num_col_ = M * J;
		model.lp_.num_row_ = M + J;
		model.lp_.a_matrix_.num_col_ = model.lp_.num_col_;
		model.lp_.a_matrix_.num_row_ = model.lp_.num_row_;

		model.lp_.offset_ = 0;
		model.lp_.sense_ = ObjSense::kMinimize;

		model.lp_.col_lower_.assign(M * J, 0);
		model.lp_.col_upper_.assign(M * J, 1);
		model.lp_.col_cost_.assign( M * J, 0);

		if (integer) {
			model.lp_.integrality_.assign(M * J, HighsVarType::kInteger);
		}

		for (int m = 0; m < M; m++) {
			for (int j = 0; j < J; j++) {
				model.lp_.col_cost_[m * J + j] = _instance.costs[m][j];
			}
		}

		// sum(x_ij) == 1
		model.lp_.row_lower_.assign(M + J, 1);
		model.lp_.row_upper_.assign(M + J, 1);

		// 0 <= wx <= C
		for (int m = 0; m < M; m++) {
			model.lp_.row_lower_[J + m] = 0;
			model.lp_.row_upper_[J + m] = _instance.capacity[m];
		}

		model.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
		model.lp_.a_matrix_.start_.resize(model.lp_.num_col_ + 1);
		model.lp_.a_matrix_.index_.resize(2 * M * J);
		model.lp_.a_matrix_.value_.resize(2 * M * J);

		for (int m = 0; m < M; m++) {
			for (int j = 0; j < J; j++) {
				model.lp_.a_matrix_.start_[m * J + j] = 2 * (m * J + j);

				// partition constraint
				model.lp_.a_matrix_.index_[2 * (m * J + j)] = j;
				model.lp_.a_matrix_.value_[2 * (m * J + j)] = 1;

				// knapsack constraint
				model.lp_.a_matrix_.index_[2 * (m * J + j) + 1] = J + m;
				model.lp_.a_matrix_.value_[2 * (m * J + j) + 1] = _instance.demands[m][j];
			}
		}
		model.lp_.a_matrix_.start_[M * J] = 2 * M * J;

		return model;
	}

public:
	// machine [job, value (1 if integer)] 
	std::vector<std::unordered_map<int, double>> get_solution() {
		auto& sol = _solution;

		std::vector<std::unordered_map<int, double>> output(_instance.machines);

		for (int m = 0; m < _instance.machines; ++m) {
			for (int j = 0; j < _instance.jobs; ++j) {
				if (sol.col_value[m * _instance.jobs + j] > 0) {
					output[m].insert({j, sol.col_value[m * _instance.jobs + j]});
				}
			}
		}

		return output;
	}
};
