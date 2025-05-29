#pragma once
#include <vector>
#include <algorithm>
#include "gap/gap_instance.h"
#include "gap/gap_pricing.h"
#include "gap/gap_compact.h"
#include "block/column_generation.h"
#include "highs/parallel/HighsParallel.h"

struct DantzigFarkas {
    GapInstance* _instance = nullptr;

    template <typename Pricer>
    void init(Pricer& pricer, PricingBlockVector<GapPricing>& pricing, gap_compact& lp) {
        _instance = pricer._instance;
    }

    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs) {
        highs::parallel::for_each(0, _instance->machines, [&](HighsInt start, HighsInt end) {
            for (int m = start; m < end; ++m) {
                reduced_costs[m] = pricing[m].optimize(duals, duals[_instance->jobs + m]);
                pricing[m].solution.push_back(_instance->jobs + m);
            }
        });

        return 0;
    }
};

struct LagrangeTemplateFarkas {
    GapInstance* _instance = nullptr;
    LagrangeTemplatePrice _template;

    template <typename Pricer>
    void init(Pricer& pricer, PricingBlockVector<GapPricing>& pricing, gap_compact& lp) {
        _instance = pricer._instance;
        _template.init(pricer._rmp, _instance);

        for (int m = 0; m < _instance->machines; ++m) {
            for (int j = 0; j < _instance->jobs; ++j) {
                double value = lp._solution.col_value[m * _instance->jobs + j];
                _template._template._template_columns[m][j] = (value > 1 - 1e-6) - (value < 1e-6);
            }
        }
    }

    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs) {
        highs::parallel::for_each(0, _instance->machines, [&](HighsInt start, HighsInt end) {
            std::vector<double> obj(_instance->jobs);

            for (int m = start; m < end; ++m) {
                reduced_costs[m] = _template.optimize_lagrangian(_template._template[m], duals, duals[_instance->jobs + m], pricing[m], _template._mu[m]);
                pricing[m].solution.push_back(_instance->jobs + m);
            }
        }, std::max(1, int(2 * _instance->machines / std::thread::hardware_concurrency())));

        return 0;
    }
};

struct TemplateFarkas {
    GapInstance* _instance = nullptr;
    TemplatePricing _template;
    std::unique_ptr<PricingBlockVector<GapPricingMIP>> _mip;

    template <typename Pricer>
    void init(Pricer& pricer, PricingBlockVector<GapPricing>& pricing, gap_compact& lp) {
        _instance = pricer._instance;
        _template.init(_instance->machines, _instance->jobs);
        _mip.reset(new PricingBlockVector<GapPricingMIP>(_instance->machines));
        _mip->init(*_instance);

        const auto& solution = lp._solution;

        for (int m = 0; m < _instance->machines; ++m) {
            for (int j = 0; j < _instance->jobs; ++j) {
                double value = solution.col_value[m * _instance->jobs + j];
                _template._template_columns[m][j] = (value > 1 - 1e-6) - (value < 1e-6);
            }
        }
    }

    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs) {
        highs::parallel::for_each(0, _instance->machines, [&](HighsInt start, HighsInt end) {
            for (int m = start; m < end; ++m) {
                reduced_costs[m] = _mip->_pricing[m].optimize_template(_template[m], duals, duals[_instance->jobs + m]);
                pricing[m].solution.swap(_mip->_pricing[m].solution);
                pricing[m].solution.push_back(_instance->jobs + m);
            }
        });

        return 0;
    }
};

