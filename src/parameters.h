#pragma once
#include <numeric>

struct Parameters {
	bool show_output = true;
	double timeout = -1;
	double gap = 0.00001;
	int nodes = 1; // std::numeric_limits<int>::max(); // max no limit, 1 root node, ...
};
