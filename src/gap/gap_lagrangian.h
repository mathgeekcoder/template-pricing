#pragma once
#include "gap_instance.h"
#include "parameters.h"
#include <vector>
#include "taskflow/taskflow.hpp"
#include "quill/CsvWriter.h"
#include "quill/core/FrontendOptions.h"
#include "block/column_generation.h"

class GapLagrangian {
	const GapInstance& instance;
	const Parameters& params;
	tf::Executor _executor;

public:
	GapLagrangian(const GapInstance& inst, const Parameters& params) : instance(inst), params(params), _executor(params.num_threads) {
		_bound = 0.0;
	}

	std::vector<double> _multipliers;
	double _bound;

	void solve(quill::CsvWriter<CsvSchema, quill::FrontendOptions>& csv_writer);
};