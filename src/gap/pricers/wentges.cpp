// Wentges code adapted from: https://github.com/fontanf/generalizedassignmentsolver
#include "wentges.h"

double norm(const std::vector<double>& a) {
	double norm = 0;
	for (auto& v : a) {
		norm += v * v;
	}
	return std::sqrt(norm);
}

double norm(const std::vector<double>& a, const std::vector<double>& b) {
	double norm = 0;
    for (int i = a.size() - 1; i >= 0; --i) {
		norm += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return std::sqrt(norm);
}

double WentgesPrice::optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs) {
    double optimal_pricing = dantzig.optimize(duals, pricing, reduced_costs);

    if (optimal_pricing < 1e-6)
        return optimal_pricing;

    for (int k = 1; ; ++k) {
        if (k > 1) {
            ++mis_price;
        }

        double alpha_cur = std::max(0.0, 1 - k * (1 - alpha) - 1e-6);
        double beta = 0;

        // stabilize duals for pricing
        if (norm(subgradient) == 0 || k > 1 || norm(stabilized_duals_for_pricing_old_columns, duals) == 0) {
            for (int row = _rmp->getNumRow() - 1; row >= 0; --row) {
                stabilized_duals_for_pricing[row] = stabilized_duals_for_pricing_old_columns[row] * alpha_cur + (1 - alpha_cur) * duals[row];
            }
        }
        else {
            // directional smoothing
			std::vector<double> pi_tilde(_rmp->getNumRow(), 0);
            std::vector<double> pi_g(_rmp->getNumRow(), 0);
            std::vector<double> rho(_rmp->getNumRow(), 0);

            // pi_tilde
            for (int row = _rmp->getNumRow() - 1; row >= 0; --row) {
                pi_tilde[row] = alpha_cur * stabilized_duals_for_pricing_old_columns[row] + (1 - alpha_cur) * duals[row];
            }

            // pi_g
            double coef_g = norm(stabilized_duals_for_pricing_old_columns, duals) / norm(subgradient);
            for (int row = _rmp->getNumRow() - 1; row >= 0; --row) {
                pi_g[row] = stabilized_duals_for_pricing_old_columns[row] + coef_g * subgradient[row];
            }

            // automatic_directional_smoothing: beta
            double dot_product = 0;
            for (int row = _rmp->getNumRow() - 1; row >= 0; --row) {
                dot_product += (duals[row] - stabilized_duals_for_pricing_old_columns[row]) * (pi_g[row] - stabilized_duals_for_pricing_old_columns[row]);
            }
            beta = dot_product / norm(stabilized_duals_for_pricing_old_columns, duals) / norm(stabilized_duals_for_pricing_old_columns, pi_g);
            beta = std::max(0.0, beta);

            // rho
            for (int row = _rmp->getNumRow() - 1; row >= 0; --row) {
                rho[row] = beta * pi_g[row] + (1 - beta) * duals[row];
            }

            // coef_sep
            double coef_sep = norm(stabilized_duals_for_pricing_old_columns, pi_tilde) / norm(stabilized_duals_for_pricing_old_columns, rho);

            for (int row = _rmp->getNumRow() - 1; row >= 0; --row) {
                stabilized_duals_for_pricing[row] = stabilized_duals_for_pricing_old_columns[row] + coef_sep * (rho[row] - stabilized_duals_for_pricing_old_columns[row]);
            }
        }

        // optimal pricing for bounds
        highs::parallel::for_each(0, _instance->machines, [&](HighsInt start, HighsInt end) {
            std::vector<double> obj = stabilized_duals_for_pricing;

            for (int m = start; m < end; ++m) {
                for (int j = 0; j < _instance->jobs; ++j) {
                    obj[j] -= _instance->profit[m][j];
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

        if (any == true || (alpha_cur == 0.0 && beta == 0.0)) {
            break;
        }
    }

    // Update alpha
    std::fill(lagrangian_constraint_values.begin(), lagrangian_constraint_values.end(), 0);
    for (int m = 0; m < _instance->machines; ++m) {
		if (reduced_costs[m] > 1e-6) {
			for (int row : pricing[m].solution) {
				++lagrangian_constraint_values[row];
			}
		}
    }

    const auto& lp = _rmp->getLp();

    // Compute subgradient at separation point
    for (int row = _rmp->getNumRow() - 1; row >= 0; --row) {
        subgradient[row] = std::min(0.0, lp.row_upper_[row] - lagrangian_constraint_values[row]) + std::max(0.0, lp.row_lower_[row] - lagrangian_constraint_values[row]);
    }

    // Adjust alpha
    if (norm(stabilized_duals_for_pricing_old_columns, stabilized_duals_for_pricing) != 0) {
        double v = 0;
        for (int row = _rmp->getNumRow() - 1; row >= 0; --row) {
            v += subgradient[row] * (stabilized_duals_for_pricing[row] - stabilized_duals_for_pricing_old_columns[row]);
        }

        alpha = v > 0 ? std::max(0.0, alpha - 0.1) : std::min(0.99, alpha + (1.0 - alpha) * 0.1);
    }

    return optimal_pricing;
}

