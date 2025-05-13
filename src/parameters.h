#pragma once
#include <numeric>

struct Parameters {
	bool show_output = true;
	double timeout = 86400;  // 1 day
	double gap = 0.00001;
	int nodes = 1; // std::numeric_limits<int>::max(); // max no limit, 1 root node, ...
	int random_seed = 0;

	// defaults are tuned for template and wentges pricing (dantzig needs more)
	double min_col_factor = 0.0;
	double max_col_factor = 0.5;
};
