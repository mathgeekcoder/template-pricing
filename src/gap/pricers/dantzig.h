#pragma once
#include "gap/gap_pricing.h"

template <typename RmpSolver>
struct DantzigPrice {
    static constexpr const char* name = "Dantzig";

    RmpSolver* _rmp = nullptr;
    GapInstance* _instance = nullptr;
    tf::Executor* _executor = nullptr;

    void init(tf::Executor* executor, RmpSolver* rmp, GapInstance* instance) {
		_executor = executor;
        _rmp = rmp;
        _instance = instance;
    }

    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs) {
		tf::Taskflow taskflow;
        tf::IndexRange range(0, _instance->machines, 1);

        taskflow.for_each_by_index(range, [&](tf::IndexRange<int> subrange) {
            std::vector<double> obj(_instance->jobs);

            for (int m = subrange.begin(); m < subrange.end(); m += subrange.step_size()) {
                for (int j = 0; j < _instance->jobs; ++j) {
                    obj[j] = duals[j] - _instance->costs[m][j];
                }

                reduced_costs[m] = pricing[m].optimize(obj, duals[_instance->jobs + m]);
                pricing[m].solution.push_back(_instance->jobs + m);
            }
        });

		_executor->run(std::move(taskflow)).wait();

        double optimal_pricing = 0.0;

        for (int m = 0; m < _instance->machines; ++m)
            optimal_pricing += reduced_costs[m] > 1e-6 ? reduced_costs[m] : 0.0;

        return optimal_pricing;
    }

    void update() {}
    void init_feasible() {}
    void debug(std::string algorithm) {}
};
