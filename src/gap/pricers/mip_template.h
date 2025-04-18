#pragma once
#include "gap/gap_pricing.h"

//
// Template-based MIP optimization
//
struct GapPricingMIP : public GapPricing {
    std::unique_ptr<Highs> highs;

    void init(uint32_t index, const GapInstance& instance);
    double optimize_template(const std::vector<double>& template_obj, const std::vector<double>& duals, double offset);
};


//
// Template-based MIP pricing
//
struct TemplatePrice {
    static constexpr const char* name = "MipTemplate";

    Highs* _rmp = nullptr;
    GapInstance* _instance = nullptr;
    TemplatePricing _template;
    std::unique_ptr<PricingBlockVector<GapPricingMIP>> _mip;

    TemplatePrice() = default;
    TemplatePrice(const TemplatePrice& copy);

    void init(Highs* rmp, GapInstance* instance);
    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs);

    void update() {
        _template.update(_rmp->getSolution(), *_rmp);
    }

    void init_feasible() {};
    void debug(std::string algorithm) {};
};


//
// Fixed Template-based MIP pricing
//
struct FixedTemplatePrice : public TemplatePrice {
    static constexpr const char* name = "FixedTemplate";

    FixedTemplatePrice() = default;
    FixedTemplatePrice(const FixedTemplatePrice& copy) : TemplatePrice(copy) { }

    // override the update, so the template is "fixed"
    void update() {};
};
