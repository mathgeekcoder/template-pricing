#pragma once
#include <numeric>

struct Parameters {
	bool show_output = true;
	double timeout = 14400;  // 4 hours
	double gap = 0.00001;
	int nodes = 1; // std::numeric_limits<int>::max(); // max no limit, 1 root node, ...
	int random_seed = 0;
	double max_col_multiplier = 1;
	double age_limit = 1;
};

