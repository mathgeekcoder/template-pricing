#pragma once
#include <numeric>
#include <string>
#include <format>

struct Parameters {
	double time_limit = 21600;  // 6 hours
	double gap = 0.00001;	 // 0.001% gap
	double max_col_multiplier = 1;
	double age_limit = 1;

	int random_seed = 0;
	int replication = 0;
	int nodes = 1; // 0 for farkas, 1 for root node (> 0 for branching, support has been removed)
	int num_threads = 1;
	bool show_output = true;
	char solver[7] = "highs"; // Options: highs, gurobi

	std::string to_json() const {
		return std::format(
			"{{timeout:{},max_col_multiplier:{},age_limit:{},random_seed:{},replication:{},num_threads:{}}}",
			time_limit,
			max_col_multiplier,
			age_limit,
			random_seed,
			replication,
			num_threads
		);
	}
};

