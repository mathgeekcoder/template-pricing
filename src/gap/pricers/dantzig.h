#pragma once
#include "gap/gap_pricing.h"

struct DantzigPrice {
    static constexpr const char* name = "Dantzig";

    Highs* _rmp = nullptr;
    GapInstance* _instance = nullptr;

    void init(Highs* rmp, GapInstance* instance) {
        _rmp = rmp;
        _instance = instance;
    }

    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs) {
        highs::parallel::for_each(0, _instance->machines, [&](HighsInt start, HighsInt end) {
            std::vector<double> obj(_instance->jobs);

            for (int m = start; m < end; ++m) {
                for (int j = 0; j < _instance->jobs; ++j) {
                    obj[j] = duals[j] - _instance->profit[m][j];
                }

                reduced_costs[m] = pricing[m].optimize(obj, duals[_instance->jobs + m]);
                pricing[m].solution.push_back(_instance->jobs + m);
            }
        });

        double optimal_pricing = 0.0;

        for (int m = 0; m < _instance->machines; ++m)
            optimal_pricing += reduced_costs[m] > 1e-6 ? reduced_costs[m] : 0.0;

        return optimal_pricing;
    }

    void update() {}
    void init_feasible() {}
    void debug(std::string algorithm) {}
};
