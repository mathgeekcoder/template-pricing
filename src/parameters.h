#pragma once
#include <numeric>

struct Parameters {
	double timeout = 21600;  // 6 hours
	double gap = 0.00001;	 // 0.001% gap
	double max_col_multiplier = 1;
	double age_limit = 1;

	int random_seed = 0;
	int replication = 0;
	int nodes = 1; // 0 for farkas, 1 for root node (> 0 for branching, support has been removed)
	int num_threads = 1;
	bool show_output = true;
	std::string column_retention = "low"; // Options: low, med, high
};

