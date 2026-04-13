// pessoa logic adapted from:
// 
// A Pessoa, R Sadykov, E Uchoa, F Vanderbeck. https://inria.hal.science/hal-01077984v4/document
// Automation and combination of linear - programming based stabilization techniques in column generation. 2014.
#pragma once
#include "gap/gap_pricing.h"
#include "dantzig.h"

template <typename RmpSolver>
struct PessoaPrice {
    static constexpr const char* name = "Pessoa";

    RmpSolver* _rmp = nullptr;
    GapInstance* _instance = nullptr;
    tf::Executor* _executor = nullptr;
    DantzigPrice<RmpSolver> dantzig;

    double best_reduced_cost = kHighsInf;
    double alpha = 0.0;
    size_t mis_price = 0;

    std::vector<double> dual_sep, dual_in;
    std::vector<double> g_sep, g_in;

    void init(tf::Executor* executor, RmpSolver* rmp, GapInstance* instance) {
        dantzig.init(executor, rmp, instance);
        _rmp = rmp;
        _instance = instance;
		_executor = executor;

        dual_sep.resize(rmp->getNumRow(), 0);
        dual_in.resize(rmp->getNumRow(), 0);

		g_sep.resize(rmp->getNumRow(), 0);
        g_in.resize(rmp->getNumRow(), 0);
    }

    void init_feasible() {
        dual_in = _rmp->getSolution().row_dual;
        dual_sep = _rmp->getSolution().row_dual;
    }

    void update() {
        dantzig.update();

        if (_rmp->getObjectiveValue() + 1e-6 < best_reduced_cost) {
            best_reduced_cost = _rmp->getObjectiveValue();
            dual_in.swap(dual_sep);
			g_in.swap(g_sep);
        }
    }

    void debug(std::string algorithm) {}
    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs);
};