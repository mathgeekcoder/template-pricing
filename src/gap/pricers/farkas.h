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
    double scale_profits = 0;
    double scale_perturb = 0;

    template <typename Pricer>
    void init(Pricer& pricer, PricingBlockVector<GapPricing>& pricing, gap_compact& lp) {
        _instance = pricer._instance;

        // get maximum profit value for scaling
        for (const auto machine_profits : _instance->profit) {
            double max_profit = *std::max_element(machine_profits.begin(), machine_profits.end());
            scale_profits = std::max(scale_profits, max_profit);
        }

        scale_profits = scale_profits > 0 ? (1 - 1e-6) / scale_profits : 0;

        // get maximum number of jobs allocated to one machine (for scaling)
        std::vector<double> ones(_instance->jobs, 1.0);

        for (int m = 0; m < _instance->machines; ++m) {
            scale_perturb = std::max(scale_perturb, pricing[m].optimize(ones, 0));
        }

        scale_perturb = scale_perturb > 0 ? (1 - 1e-6) / scale_perturb : 0;
    }

    double scale(const std::vector<double>& duals) {
        // the duals (from extreme ray) can be sparse, which leads to volatile results
        // want to perturb with "small enough" values to improve convergence

        // find smallest non-zero value
        double min_val = kHighsInf;

        for (int j = 0; j < _instance->jobs; ++j) {
            auto value = std::abs(duals[j]);
            if (value > 0 && value < min_val) {
                min_val = value;
            }
        }

        // want to add small enough values so that sum(perturbations) < min_val
        return scale_perturb * min_val;
    }

    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs) {
        double multiplier = scale(duals);

        highs::parallel::for_each(0, _instance->machines, [&](HighsInt start, HighsInt end) {
            std::vector<double> obj(_instance->jobs);

            for (int m = start; m < end; ++m) {
                for (int j = 0; j < _instance->jobs; ++j) {
                    obj[j] = duals[j] + multiplier * (1 - scale_profits * _instance->profit[m][j]);
                }

                reduced_costs[m] = pricing[m].optimize(obj, duals[_instance->jobs + m]);
                pricing[m].solution.push_back(_instance->jobs + m);
            }
        });

        return 0;
    }
};

struct WentgesTemplateFarkas : DantzigFarkas {
    WentgesTemplatePrice _template;

    template <typename Pricer>
    void init(Pricer& pricer, PricingBlockVector<GapPricing>& pricing, gap_compact& lp) {
        DantzigFarkas::init(pricer, pricing, lp);
        _template.init(pricer._rmp, _instance);

        for (int m = 0; m < _instance->machines; ++m) {
            for (int j = 0; j < _instance->jobs; ++j) {
                double value = lp._solution.col_value[m * _instance->jobs + j];
                _template._template._template_columns[m][j] = (value > 1 - 1e-6) - (value < 1e-6);
            }
        }
    }

    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs) {
        double multiplier = scale(duals);

        highs::parallel::for_each(0, _instance->machines, [&](HighsInt start, HighsInt end) {
            std::vector<double> obj(_instance->jobs);

            for (int m = start; m < end; ++m) {
                for (int j = 0; j < _instance->jobs; ++j) {
                    obj[j] = duals[j] - multiplier * scale_profits * _instance->profit[m][j];
                }

                reduced_costs[m] = _template.optimize_lagrangian(_template._template[m], obj, duals[_instance->jobs + m], pricing[m], _template._mu[m]);
                pricing[m].solution.push_back(_instance->jobs + m);
            }
        }, std::max(1, int(2 * _instance->machines / std::thread::hardware_concurrency())));

        return 0;
    }
};

struct TemplateFarkas : DantzigFarkas {
    TemplatePricing _template;
    std::unique_ptr<PricingBlockVector<GapPricingMIP>> _mip;

    template <typename Pricer>
    void init(Pricer& pricer, PricingBlockVector<GapPricing>& pricing, gap_compact& lp) {
        DantzigFarkas::init(pricer, pricing, lp);

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
        double multiplier = scale(duals);

        highs::parallel::for_each(0, _instance->machines, [&](HighsInt start, HighsInt end) {
            std::vector<double> obj(_instance->jobs);

            for (int m = start; m < end; ++m) {
                for (int j = 0; j < _instance->jobs; ++j) {
                    obj[j] = duals[j] - multiplier * scale_profits * _instance->profit[m][j];
                }

                reduced_costs[m] = _mip->_pricing[m].optimize_template(_template[m], obj, duals[_instance->jobs + m]);
                pricing[m].solution.swap(_mip->_pricing[m].solution);
                pricing[m].solution.push_back(_instance->jobs + m);
            }
        });

        return 0;
    }
};

struct FixedTemplateFarkas {
    GapInstance* _instance = nullptr;
    TemplatePricing _template;
    std::unique_ptr<PricingBlockVector<GapPricingMIP>> _mip;

    template <typename Pricer>
    void init(Pricer& pricer, PricingBlockVector<GapPricing>& pricing, gap_compact& lp) {
        _instance = pricer._instance;
        _mip.reset(new PricingBlockVector<GapPricingMIP>(_instance->machines));
        _mip->init(*_instance);
    }

    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs) {
        highs::parallel::for_each(0, _instance->machines, [&](HighsInt start, HighsInt end) {
            for (int m = start; m < end; ++m) {
                double obj = _mip->_pricing[m].optimize_template(_template[m], duals, duals[_instance->jobs + m]);
                pricing[m].solution.swap(_mip->_pricing[m].solution);
                pricing[m].solution.push_back(_instance->jobs + m);
                reduced_costs[m] = obj != -kHighsInf ? 1 : -kHighsInf;
            }
        });

        double feasible = 0;
        for (int m = 0; m < _instance->machines; ++m) {
            feasible = std::min(feasible, reduced_costs[m]);
        }

        return feasible;
    }
};
