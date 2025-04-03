#include "wentges_template.h"

double WentgesTemplatePrice::optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs, bool update_duals) {
    std::vector<double> rc(_instance->machines, 0);

    highs::parallel::for_each(0, _instance->machines, [&](HighsInt start, HighsInt end) {
        for (int m = start; m < end; ++m) {
            std::vector<double> obj = duals;

            if (update_duals) {
                for (int j = 0; j < _instance->jobs; ++j) {
                    obj[j] -= _instance->profit[m][j];
                }
            }

            // optimal pricing for bounds
            double offset = duals[_instance->jobs + m];
            rc[m] = pricing[m].optimize(obj, offset);
            reduced_costs[m] = rc[m];

            if (reduced_costs[m] > 1e-6) {
                reduced_costs[m] = optimize_lagrangian(_template[m], obj, offset, pricing[m], _mu[m]);
            }

            pricing[m].solution.push_back(_instance->jobs + m);
        }
    });

    double optimal_pricing = 0.0;

    for (int m = 0; m < _instance->machines; ++m)
        optimal_pricing += rc[m] > 1e-6 ? rc[m] : 0.0;

    return optimal_pricing;
}

double WentgesTemplatePrice::optimize_lagrangian(const std::vector<double>& template_obj, const std::vector<double>& obj, double offset, GapPricing& pricer, double& hi_mu) {
    double ub = _instance->jobs;
    double best = -_instance->jobs;
    double best_rc = 0;

    std::vector<HighsInt> best_solution;
    std::vector<double> tmpProfit(_instance->jobs);
    double lo_mu = 0;

    if (hi_mu == 0) {
        hi_mu = 0.5;
    }

    // find first feasible hi_mu
    while (true) {
        for (int j = 0; j < _instance->jobs; ++j) {
            tmpProfit[j] = template_obj[j] + hi_mu * obj[j];
        }

        auto opt = pricer.optimize(tmpProfit, hi_mu * offset);
        ub = std::min(ub, std::floor(opt));

        double rc = sum(pricer.solution, offset, obj);

        if (rc > 1e-6) {
            best_rc = rc;
            best = sum(pricer.solution, 0.0, template_obj);
            break;
        }

        lo_mu = hi_mu;
        hi_mu *= 2;
    }

    best_solution.swap(pricer.solution);

	// binary search for best mu
    while (hi_mu - lo_mu > 1e-6 && best < ub) {
        double mu = 0.5 * (hi_mu + lo_mu);

        for (int i = 0; i < _instance->jobs; ++i) {
            tmpProfit[i] = template_obj[i] + mu * obj[i];
        }

        auto opt = pricer.optimize(tmpProfit, mu * offset);
        ub = std::min(ub, std::floor(opt));

        double rc = sum(pricer.solution, offset, obj);

        if (rc <= 1e-6) {
            lo_mu = mu;
        }
        else {
            hi_mu = mu;
            double tmpl = sum(pricer.solution, 0.0, template_obj);

            if (best < tmpl || best == tmpl && best_rc < rc) {
                best = tmpl;
                best_rc = rc;
                best_solution.swap(pricer.solution);
            }
        }
    }

    pricer.solution.swap(best_solution);
    return best_rc;
}
