#pragma once
#include "gap/gap_pricing.h"

struct LagrangeTemplatePrice {
    static constexpr const char* name = "LagrangeTemplate";

    Highs* _rmp = nullptr;
    GapInstance* _instance = nullptr;
    TemplatePricing _template;
    std::vector<double> _mu;

    void init(Highs* rmp, GapInstance* instance) {
        _rmp = rmp;
        _instance = instance;
        _template.init(instance->machines, instance->jobs);
        _mu.resize(instance->machines, 1);
    }

    void update() {
        _template.update(_rmp->getSolution(), *_rmp);
    }

    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs, bool update_duals = true);
    double optimize_lagrangian(const std::vector<double>& template_obj, const std::vector<double>& obj, double offset, GapPricing& pricer, double& mu);

    void init_feasible() {}
    void debug(std::string algorithm) {}
};
