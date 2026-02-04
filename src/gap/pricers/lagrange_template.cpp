#include "lagrange_template.h"
#include "taskflow/taskflow.hpp"
#include "taskflow/algorithm/for_each.hpp"

template <typename RmpSolver>
double LagrangeTemplatePrice<RmpSolver>::optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs, bool update_duals) {
    std::vector<double> rc(_instance->machines, 0);

    tf::Taskflow taskflow;
	tf::IndexRange range(0, _instance->machines, 1);

    taskflow.for_each_by_index(range, [&](tf::IndexRange<int> subrange) {
        std::vector<double> obj(_instance->jobs);

        for (int m = subrange.begin(); m < subrange.end(); m += subrange.step_size()) {
            if (update_duals) {
                for (int j = 0; j < _instance->jobs; ++j) {
                    obj[j] = duals[j] - _instance->costs[m][j];
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

	_executor->run(std::move(taskflow)).wait();

    double optimal_pricing = 0.0;

    for (int m = 0; m < _instance->machines; ++m)
        optimal_pricing += rc[m] > 1e-6 ? rc[m] : 0.0;

    //for (int m = 0; m < _instance->machines; ++m) {
    //    std::cout << (rc[m] > 1e-6 ? _mu[m] : -1) << " ";
    //}
    //for (int m = 0; m < _instance->machines; ++m) {
    //    std::cout << (rc[m] > 1e-6 ? _counts[m] : -1) << " ";
    //}
    //std::cout << " " << optimal_pricing << std::endl;

    return optimal_pricing;
}

template <typename RmpSolver>
double LagrangeTemplatePrice<RmpSolver>::optimize_lagrangian(const std::vector<double>& template_obj, const std::vector<double>& obj, double offset, GapPricing& pricer, double& hi_mu) {
    double ub = _instance->jobs;
    double best = -_instance->jobs;
    double best_rc = 0;

    std::vector<HighsInt> best_solution;
    std::vector<double> tmpProfit(_instance->jobs);
    double lo_mu = 0;

    if (hi_mu == 0) {
        hi_mu = 0.5;
    }

    int machine = &hi_mu - &_mu[0];
    _counts[machine] = 0;

    // find first feasible hi_mu
    while (hi_mu < kHighsInf) {
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
		_counts[machine]++;
    }

	// shouldn't occur since calling functions perform checks, but just in case
    if (hi_mu == kHighsInf) {
        hi_mu = 0.5;
        return -kHighsInf;
    }

    best_solution.swap(pricer.solution);

	// binary search for best mu
//    while (hi_mu - lo_mu > 1e-6 && best < ub) {
    while (hi_mu > 1.001 * lo_mu && best < ub) {
        double mu = 0.5 * (hi_mu + lo_mu);

        for (int i = 0; i < _instance->jobs; ++i) {
            tmpProfit[i] = template_obj[i] + mu * obj[i];
        }

        // fast approximate - can sometimes find better solutions
        pricer.approx(tmpProfit, mu * offset);
        double approx_rc = sum(pricer.solution, offset, obj);

        if (approx_rc > 1e-6) {
            double tmpl = sum(pricer.solution, 0.0, template_obj);

            if (best < tmpl || best == tmpl && best_rc < approx_rc) {
                best = tmpl;
                best_rc = approx_rc;
                best_solution.swap(pricer.solution);
            }
        }

        // optimal
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
        _counts[machine]++;
    }

    pricer.solution.swap(best_solution);
    return best_rc;
}

template class LagrangeTemplatePrice<Highs>;

#ifdef SUPPORT_GUROBI

template class LagrangeTemplatePrice<GurobiHighs>;

#endif