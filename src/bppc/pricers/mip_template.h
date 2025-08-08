#pragma once
#include "bppc/bppc_pricing.h"

//
// Template-based MIP optimization
//
struct BppcPricingMIP : public BppcPricing {
    void init(uint32_t index, const BppcInstance& instance);
    double optimize_template(const std::vector<double>& template_obj, const std::vector<double>& duals, double offset);
};


//
// Template-based MIP pricing
//
struct BppcTemplatePrice {
    static constexpr const char* name = "MipTemplate";

    Highs* _rmp = nullptr;
    BppcInstance* _instance = nullptr;
    TemplatePricing _template;
    std::unique_ptr<PricingBlockVector<BppcPricingMIP>> _mip;
	int _machines = 1;
	bool _farkas = true; // whether to use Farkas pricing

    BppcTemplatePrice() = default;
    BppcTemplatePrice(const BppcTemplatePrice& copy);

    void init(Highs* rmp, BppcInstance* instance);
    double optimize(const std::vector<double>& duals, PricingBlockVector<BppcPricing>& pricing, std::vector<double>& reduced_costs);

    void update();
    void init_feasible() {
        _machines = std::floor(_rmp->getObjectiveValue());

        _mip.reset(new PricingBlockVector<BppcPricingMIP>(_machines));
        _mip->init(*_instance);
        _farkas = false;
    };
};
