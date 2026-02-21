#pragma once
#include "gap_instance.h"
#include "block/column_generation.h"
#include "highs/Highs.h"
#include "extern/scip/scip_knapsack.h"
#include "taskflow/taskflow.hpp"
#include "taskflow/algorithm/for_each.hpp"

struct GapPricing {
    size_t _machine = 0;
    const GapInstance* _instance = nullptr;
    std::vector<int> solution;
	std::vector<double> scip_tmp;  // temporary storage for faster SCIP solve

    void init(uint32_t index, const GapInstance& instance) {
        _instance = &instance;
        _machine = index;
    }

    double optimize(const std::vector<double>& obj, double offset) {
        double opt;

        int solution_size = 0;
        solution.resize(_instance->jobs);
        bool success = SCIPsolveKnapsackExactly(_instance->jobs, _instance->demands[_machine].data(), obj.data(), _instance->capacity[_machine], solution.data(), &solution_size, &opt, scip_tmp);
        solution.resize(solution_size);

        if (!success) {
			throw std::runtime_error("Error in SCIP Knapsack");
   		}

        return opt + offset;
    }

    double approx(const std::vector<double>& obj, double offset) {
        double approx_opt;

		// need to copy, as SCIPsolveKnapsackApproximately modifies arrays
        std::vector<int> weights = _instance->demands[_machine];
        std::vector<double> tmp_obj = obj;
        std::vector<int> index(_instance->jobs);
        std::iota(index.begin(), index.end(), 0);

        int solution_size = 0;
        solution.resize(_instance->jobs);

        SCIPsolveKnapsackApproximately(_instance->jobs, weights.data(), tmp_obj.data(), _instance->capacity[_machine],
            index.data(), solution.data(), nullptr, &solution_size, nullptr, &approx_opt);
        solution.resize(solution_size);
        return approx_opt + offset;
	}
};
