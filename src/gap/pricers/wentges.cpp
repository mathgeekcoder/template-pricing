#include "wentges.h"

double norm(const std::vector<double>& a) {
	double norm = 0;
	for (const auto& v : a) {
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

double WentgesPrice::optimize(const std::vector<double>& dual_out, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs) {
    double optimal_pricing = dantzig.optimize(dual_out, pricing, reduced_costs);

    if (optimal_pricing < 1e-6)
        return optimal_pricing;

    for (int k = 1; ; ++k) {
        if (k > 1) {
            ++mis_price;
        }

        double alpha_tilde = std::max(0.0, 1 - k * (1 - alpha));
        double beta = 0;

        // stabilize duals for pricing
        if (norm(g_in) == 0 || k > 1) {
            for (int row = _rmp->getNumRow() - 1; row >= 0; --row) {
                dual_sep[row] = alpha_tilde * dual_in[row] + (1 - alpha_tilde) * dual_out[row];
            }
        }
        else {
			// step 1. pi_tilde
            std::vector<double> pi_tilde(_rmp->getNumRow());

			for (int row = _rmp->getNumRow() - 1; row >= 0; --row) {
                pi_tilde[row] = alpha_tilde * dual_in[row] + (1 - alpha_tilde) * dual_out[row];
			}

            // step 2. pi_g
            std::vector<double> pi_g(_rmp->getNumRow());
			double g = norm(dual_out, dual_in) / norm(g_in);

            for (int row = _rmp->getNumRow() - 1; row >= 0; --row) {
                pi_g[row] = dual_in[row] + g * g_in[row];
            }

            // step 2.5. 
            // beta = cos(angle between (dual_out - dual_in) and (pi_g - dual_in))
			double dp = 0;
			for (int row = _rmp->getNumRow() - 1; row >= 0; --row) {
                dp += (dual_out[row] - dual_in[row]) * (pi_g[row] - dual_in[row]);
			}
			beta = dp / (norm(dual_out, dual_in) * norm(pi_g, dual_in));  // (0, 1]

            // step 3. rho
			std::vector<double> rho(_rmp->getNumRow());
            for (int row = _rmp->getNumRow() - 1; row >= 0; --row) {
				rho[row] = beta * pi_g[row] + (1 - beta) * dual_out[row];
            }

			// step 4. dual_sep
            double coeff = norm(pi_tilde, dual_in) / norm(rho, dual_in);

            for (int row = _rmp->getNumRow() - 1; row >= 0; --row) {
                dual_sep[row] = std::max(0.0, dual_in[row] + coeff * (rho[row] - dual_in[row]));
            }
        }

        // optimal pricing for bounds
        highs::parallel::for_each(0, _instance->machines, [&](HighsInt start, HighsInt end) {
            std::vector<double> obj = dual_sep;

            for (int m = start; m < end; ++m) {
                for (int j = 0; j < _instance->jobs; ++j) {
                    obj[j] -= _instance->profit[m][j];
                }

                pricing[m].optimize(obj, dual_sep[_instance->jobs + m]);

                reduced_costs[m] = dual_out[_instance->jobs + m];
                for (auto j : pricing[m].solution) {
                    reduced_costs[m] += dual_out[j] - _instance->profit[m][j];
                }

                pricing[m].solution.push_back(_instance->jobs + m);
            }
        });

        bool any = false;

        for (int m = 0; m < _instance->machines && any == false; ++m) {
            any = (reduced_costs[m] > 1e-6);
        }

        if (any == true || (alpha_tilde == 0.0 && beta == 0.0)) {
            break;
        }
    }

    // Update subgradient and alpha
    std::vector<int> job_count(_instance->jobs + _instance->machines, 0);
    for (int m = 0; m < _instance->machines; ++m) {
        if (reduced_costs[m] > 1e-6) {
            for (int j : pricing[m].solution) {
                ++job_count[j];
            }
        }
    }

    // Compute subgradient at separation point
    // even though RMP is a cover (assumed here), we want to find the subgradient assuming a partition
    for (int j = 0; j < _instance->jobs; ++j) {
        g_sep[j] = job_count[j] - 1; // violated job constraint
    }

    // Adjust alpha
    double v = 0;
    for (int row = _rmp->getNumRow() - 1; row >= 0; --row) {
        v += g_sep[row] * (dual_out[row] - dual_in[row]);
    }

    alpha = v > 0 ? std::min(0.99, alpha + (1.0 - alpha) * 0.1) : std::max(0.0, alpha - 0.1);

    return optimal_pricing;
}

