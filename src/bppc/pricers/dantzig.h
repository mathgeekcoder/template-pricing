#pragma once
#include "bppc/bppc_pricing.h"

struct BppcDantzigPrice {
    static constexpr const char* name = "Dantzig";

    Highs* _rmp = nullptr;
    BppcInstance* _instance = nullptr;
    TemplatePricing _template;
    int _machines = 1;
    bool _farkas = true; // whether to use Farkas pricing


    void init(Highs* rmp, BppcInstance* instance) {
        _rmp = rmp;
        _instance = instance;
    }

    double optimize(const std::vector<double>& duals, PricingBlockVector<BppcPricing>& pricing, std::vector<double>& reduced_costs) {
        reduced_costs[0] = pricing[0].optimize(duals, -1);
        return reduced_costs[0] > 1e-6 ? reduced_costs[0] : 0.0;
    }

    void update() {}
    void init_feasible() {}
};
