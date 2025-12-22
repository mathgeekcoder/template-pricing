#pragma once
#include "gap/gap_pricing.h"

template <typename RmpSolver>
struct LagrangeTemplatePrice {
    static constexpr const char* name = "LagrangeTemplate";

    RmpSolver* _rmp = nullptr;
    GapInstance* _instance = nullptr;
    TemplatePricing _template;
	tf::Executor* _executor = nullptr;
    std::vector<double> _mu;
    std::vector<int> _counts;

    void init(tf::Executor* executor, RmpSolver* rmp, GapInstance* instance) {
        _rmp = rmp;
        _instance = instance;
        _template.init(instance->machines, instance->jobs);
        _mu.resize(instance->machines, 1);
		_executor = executor;
		_counts.resize(instance->machines, 0);
    }

    void update() {
        _template.update(_rmp->getSolution(), *_rmp);
    }

    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs, bool update_duals = true);
    double optimize_lagrangian(const std::vector<double>& template_obj, const std::vector<double>& obj, double offset, GapPricing& pricer, double& mu);

    void init_feasible() {}
    void debug(std::string algorithm) {}
};
