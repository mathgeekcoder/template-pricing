#include "wentges.h"

double WentgesPrice::optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs) {
    double optimal_pricing = dantzig.optimize(duals, pricing, reduced_costs);

    for (int k = 1; alpha > 0; ++k) {
        if (k > 1) {
            ++mis_price;
            alpha = std::max(0.0, 1 - k * (1 - alpha) - 1e-6);
        }

        // stabilize duals for pricing
        for (int row = _rmp->getNumRow() - 1; row >= 0; --row) {
            stabilized_duals_for_pricing[row] = stabilized_duals_for_pricing_old_columns[row] * alpha + (1 - alpha) * duals[row];
        }

        // optimal pricing for bounds
        highs::parallel::for_each(0, _instance->machines, [&](HighsInt start, HighsInt end) {
            std::vector<double> obj(_instance->jobs);

            for (int m = start; m < end; ++m) {
                for (int j = 0; j < _instance->jobs; ++j) {
                    obj[j] = stabilized_duals_for_pricing[j] - _instance->profit[m][j];
                }

                pricing[m].optimize(obj, stabilized_duals_for_pricing[_instance->jobs + m]);

                reduced_costs[m] = duals[_instance->jobs + m];
                for (auto j : pricing[m].solution) {
                    reduced_costs[m] += duals[j] - _instance->profit[m][j];
                }

                pricing[m].solution.push_back(_instance->jobs + m);
            }
            });

        bool any = false;

        for (int m = 0; m < _instance->machines && any == false; ++m) {
            any = (reduced_costs[m] > 1e-6);
        }

        if (any == true) {
            break;
        }
    }

    // Update alpha
    std::fill(lagrangian_constraint_values.begin(), lagrangian_constraint_values.end(), 0);
    for (int m = 0; m < _instance->machines; ++m) {
        for (int row : pricing[m].solution) {
            ++lagrangian_constraint_values[row];
        }
    }

    // Compute subgradient at separation point.
    for (int row = _rmp->getNumRow() - 1; row >= 0; --row) {
        subgradient[row] = std::min(0.0, 1 - lagrangian_constraint_values[row]) + std::max(0.0, 1 - lagrangian_constraint_values[row]);
    }

    // Adjust alpha
    double norm2 = 0;
    for (int row = _rmp->getNumRow() - 1; row >= 0; --row) {
        norm2 += std::abs(stabilized_duals_for_pricing_old_columns[row] - duals[row]);
    }

    if (norm2 != 0) {
        double v = 0;
        for (int row = _rmp->getNumRow() - 1; row >= 0; --row) {
            v += subgradient[row] * (duals[row] - stabilized_duals_for_pricing_old_columns[row]);
        }

        alpha = v > 0 ? std::max(0.0, alpha - 0.1) : std::min(0.99, alpha + (1.0 - alpha) * 0.1);
    }

    return optimal_pricing;
}

