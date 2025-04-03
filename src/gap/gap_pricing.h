#pragma once
#include "gap_instance.h"
#include "extern/scip/scip_knapsack.h"
#include "block/column_generation.h"
#include "highs/Highs.h"
#include "highs/parallel/HighsParallel.h"

struct GapPricing {
    size_t _machine = 0;
    const GapInstance* _instance = nullptr;
    std::vector<int> solution;
    std::vector<bool> lb_bound;

    void init(uint32_t index, const GapInstance& instance) {
        _instance = &instance;
        _machine = index;
    }

    double optimize(const std::vector<double>& obj, double offset) {
        double scip_opt;
        int solution_size = 0;
        solution.resize(_instance->jobs);
        SCIPsolveKnapsackExactly(_instance->jobs, _instance->demands[_machine].data(), obj.data(), _instance->capacity[_machine], solution.data(), &solution_size, &scip_opt);
        solution.resize(solution_size);

        return scip_opt + offset;
    }
};
