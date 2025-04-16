#pragma once
#include "gap/gap_pricing.h"
#include "dantzig.h"

struct WentgesPrice {
    static constexpr const char* name = "Wentges";

    Highs* _rmp = nullptr;
    GapInstance* _instance = nullptr;
    DantzigPrice dantzig;

    double best_reduced_cost = kHighsInf;
    double alpha = 0.0;
    size_t mis_price = 0;

    std::vector<double> stabilized_duals_for_pricing_old_columns;
    std::vector<double> stabilized_duals_for_pricing;

    std::vector<double> lagrangian_constraint_values;
    std::vector<double> subgradient;

    void init(Highs* rmp, GapInstance* instance) {
        dantzig.init(rmp, instance);
        _rmp = rmp;
        _instance = instance;

        stabilized_duals_for_pricing_old_columns.resize(rmp->getNumRow(), 0);
        stabilized_duals_for_pricing.resize(rmp->getNumRow(), 0);
        lagrangian_constraint_values.resize(rmp->getNumRow(), 0);
        subgradient.resize(rmp->getNumRow(), 0);
    }

    void init_feasible() {
        stabilized_duals_for_pricing = _rmp->getSolution().row_dual;
    }

    void update() {
        if (_rmp->getObjectiveValue() < best_reduced_cost) {
            best_reduced_cost = _rmp->getObjectiveValue();
            stabilized_duals_for_pricing_old_columns = stabilized_duals_for_pricing;
        }
    }

    void debug(std::string algorithm) {}
    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs);
};