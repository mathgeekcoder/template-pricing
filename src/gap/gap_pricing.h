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
	std::vector<bool> lb_bound;    // used for GUB branching
	std::vector<double> scip_tmp;  // temporary storage for faster SCIP solve

    void init(uint32_t index, const GapInstance& instance) {
        _instance = &instance;
        _machine = index;
    }

    double optimize(const std::vector<double>& obj, double offset) {
        double opt;

        int solution_size = 0;
        solution.resize(_instance->jobs);
        SCIPsolveKnapsackExactly(_instance->jobs, _instance->demands[_machine].data(), obj.data(), _instance->capacity[_machine], solution.data(), &solution_size, &opt, scip_tmp);
        solution.resize(solution_size);

        return opt + offset;
    }
};
