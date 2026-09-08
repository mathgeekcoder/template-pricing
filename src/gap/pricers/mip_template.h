#pragma once
#include "gap/gap_pricing.h"

//
// Template-based MIP optimization
//
template <typename RmpSolver>
struct GapPricingMIP : public GapPricing {
    std::unique_ptr<RmpSolver> highs;

    void init(uint32_t index, const GapInstance& instance);
    double optimize_template(const std::vector<double>& template_obj, const std::vector<double>& duals, double offset);
};


//
// Template-based MIP pricing
//
template <typename RmpSolver>
struct TemplatePrice {
    static constexpr const char* name = "MipTemplate";

    RmpSolver* _rmp = nullptr;
    GapInstance* _instance = nullptr;
	tf::Executor* _executor = nullptr;
    TemplatePricing _template;
    std::unique_ptr<PricingBlockVector<GapPricingMIP<RmpSolver>>> _mip;

    void init(tf::Executor* executor, RmpSolver* rmp, GapInstance* instance);
    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs);

    void update() {
        _template.update(_rmp->getSolution(), *_rmp);
    }

    void init_feasible() {};
    void debug(std::string algorithm) {};
};
