#pragma once
#include <vector>
#include <algorithm>
#include "gap/gap_instance.h"
#include "gap/gap_pricing.h"
#include "gap/gap_compact.h"
#include "block/column_generation.h"

template <typename RmpSolver>
struct DantzigFarkas {
    static constexpr const char* name = "DantzigFarkas";

    GapInstance* _instance = nullptr;
	tf::Executor* _executor = nullptr;

    template <typename Pricer>
    void init(tf::Executor* executor, Pricer& pricer, PricingBlockVector<GapPricing>& pricing, GapCompact<RmpSolver>& lp) {
        _instance = pricer._instance;
        _executor = executor;
    }

    void update(RmpSolver* rmp) { }

    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs) {
		tf::Taskflow taskflow;

        taskflow.for_each_index(0, _instance->machines, 1, [&](int m) {
            reduced_costs[m] = pricing[m].optimize(duals, duals[_instance->jobs + m]);
            pricing[m].solution.push_back(_instance->jobs + m);
        });

		_executor->run(std::move(taskflow)).wait();
        return 0;
    }
};

template <typename RmpSolver>
struct LagrangeTemplateFarkas {
    static constexpr const char* name = "LTFarkas";

    GapInstance* _instance = nullptr;
    tf::Executor* _executor = nullptr;
    LagrangeTemplatePrice<RmpSolver> _template;

    template <typename Pricer>
    void init(tf::Executor* executor, Pricer& pricer, PricingBlockVector<GapPricing>& pricing, GapCompact<RmpSolver>& lp) {
        _instance = pricer._instance;
        _template.init(executor, pricer._rmp, _instance);
        _executor = executor;

        for (int m = 0; m < _instance->machines; ++m) {
            for (int j = 0; j < _instance->jobs; ++j) {
                double value = lp._solution.col_value[m * _instance->jobs + j];
                _template._template._template_columns[m][j] = (value > 1 - 1e-6) - (value < 1e-6);
            }
        }
    }

    // Update the template but ignore the dummy variables
    void update(RmpSolver* rmp) {
        for (int block = 0; block < _instance->machines; ++block) {
            std::fill(_template._template._template_columns[block].begin(), _template._template._template_columns[block].end(), 0.0);
        }

        // assumes lp colwise
		const auto& solution = rmp->getSolution();

        for (size_t i = _instance->jobs, end = solution.col_value.size(); i < end; ++i) {
            if (solution.col_value[i] > 1e-6) {
                const auto& col = get_column(*rmp, i);
                auto end = std::prev(col.end());  // assume last element is the block index
                auto& template_block = _template._template._template_columns[*end - _instance->jobs];

                for (auto it = col.begin(); it != end; ++it) {
                    template_block[*it] += solution.col_value[i] / solution.row_value[*it];
                }
            }
        }

        for (auto& b : _template._template._template_columns) {
            for (auto& c : b) {
                c = (c > 1 - 1e-6) - (c < 1e-6);
            }
        }
    }

    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs) {
        tf::Taskflow taskflow;

        taskflow.for_each_index(0, _instance->machines, 1, [&](int m) {
            reduced_costs[m] = pricing[m].optimize(duals, duals[_instance->jobs + m]);

			// if reduce cost is non-positive, lagrangian will require hi_mu to be infinite
            if (reduced_costs[m] > 1e-6) {
                reduced_costs[m] = _template.optimize_lagrangian(_template._template[m], duals, duals[_instance->jobs + m], pricing[m], _template._mu[m]);
            }

            pricing[m].solution.push_back(_instance->jobs + m);
        });

        _executor->run(std::move(taskflow)).wait();
        return 0;
    }
};

template <typename RmpSolver>
struct TemplateFarkas {
    static constexpr const char* name = "MTFarkas";

    GapInstance* _instance = nullptr;
    TemplatePricing _template;
	tf::Executor* _executor = nullptr;
    std::unique_ptr<PricingBlockVector<GapPricingMIP<RmpSolver>>> _mip;

    template <typename Pricer>
    void init(tf::Executor* executor, Pricer& pricer, PricingBlockVector<GapPricing>& pricing, GapCompact<RmpSolver>& lp) {
        _instance = pricer._instance;
		_executor = executor;
        _template.init(_instance->machines, _instance->jobs);
        _mip.reset(new PricingBlockVector<GapPricingMIP<RmpSolver>>(_instance->machines));
        _mip->init(*_instance);

        const auto& solution = lp._solution;

        for (int m = 0; m < _instance->machines; ++m) {
            for (int j = 0; j < _instance->jobs; ++j) {
                double value = solution.col_value[m * _instance->jobs + j];
                _template._template_columns[m][j] = (value > 1 - 1e-6) - (value < 1e-6);
            }
        }
    }

    // Update the template but ignore the dummy variables
    void update(RmpSolver* rmp) {
        for (int block = 0; block < _instance->machines; ++block) {
            std::fill(_template._template_columns[block].begin(), _template._template_columns[block].end(), 0.0);
        }

        // assumes lp colwise
        const auto& solution = rmp->getSolution();

        for (size_t i = _instance->jobs, end = solution.col_value.size(); i < end; ++i) {
            if (solution.col_value[i] > 1e-6) {
                const auto& col = get_column(*rmp, i);
                auto end = std::prev(col.end());  // assume last element is the block index
                auto& template_block = _template._template_columns[*end - _instance->jobs];

                for (auto it = col.begin(); it != end; ++it) {
                    template_block[*it] += solution.col_value[i] / solution.row_value[*it];
                }
            }
        }

        for (auto& b : _template._template_columns) {
            for (auto& c : b) {
                c = (c > 1 - 1e-6) - (c < 1e-6);
            }
        }
    }

    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs) {
        tf::Taskflow taskflow;

        taskflow.for_each_index(0, _instance->machines, 1, [&](int m) {
            reduced_costs[m] = _mip->_pricing[m].optimize_template(_template[m], duals, duals[_instance->jobs + m]);
            pricing[m].solution.swap(_mip->_pricing[m].solution);
            pricing[m].solution.push_back(_instance->jobs + m);
        });

        _executor->run(std::move(taskflow)).wait();
        return 0;
    }
};

