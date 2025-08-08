#include "mip_template.h"
#include <numeric> // Add this include for std::accumulate

void BppcPricingMIP::init(uint32_t index, const BppcInstance& instance) {
    BppcPricing::init(index, instance);

	// want to add 2 extra rows:
    //   eps <= reduced costs <= inf
    //    lb <= template costs <= ub

	std::vector<int> items(instance.items, 0);
	std::vector<double> values(instance.items, 1);
	std::iota(items.begin(), items.end(), 0);

    highs->addRow(0, kHighsInf, instance.items, items.data(), values.data());
    highs->addRow(0, kHighsInf, instance.items, items.data(), values.data());
}

double BppcPricingMIP::optimize_template(const std::vector<double>& template_obj, const std::vector<double>& duals, double offset) {
    // heuristic dual & template

	// can assume that we've already checked if a solution exists

	// since we modify the shared highs model, we need to ensure that the model is reset for dantzig pricing
    int row = highs->getNumRow() - 2;
	//highs->changeObjectiveOffset(0);
 //   highs->changeRowBounds(row, -kHighsInf, kHighsInf);
 //   highs->changeRowBounds(row + 1, -kHighsInf, kHighsInf);

 //   double tmp = optimize(duals, offset);
 //   if (tmp <= 1e-6) {
 //       return tmp;
 //   }

    highs->changeColsCost(0, _instance->items - 1, template_obj.data());
    highs->changeRowBounds(row, -offset, kHighsInf);
    highs->changeRowBounds(row + 1, -_instance->items, kHighsInf);

    for (int j = 0; j < _instance->items; ++j)
        highs->changeCoeff(row, j, duals[j]);

    for (int j = 0; j < _instance->items; ++j)
        highs->changeCoeff(row + 1, j, template_obj[j]);

	//highs->writeModel("sub" + std::to_string(_machine) + ".mps");
    highs->run();
    double ht = highs->getObjectiveValue(), hd = offset;

    // switch objective, add new constraint
    HighsStatus status = highs->changeColsCost(0, _instance->items - 1, duals.data());
    highs->changeRowBounds(row + 1, std::ceil(ht - 0.5), kHighsInf);
    highs->changeObjectiveOffset(offset);
    status = highs->run();

    // ensure good reduced costs, prevent infinite loop
    while (ht > -_instance->items && highs->getObjectiveValue() <= 1e-6) {
        --ht;
        highs->changeRowBounds(row + 1, std::ceil(ht - 0.5), kHighsInf);
        status = highs->run();
    }

    if (ht <= -_instance->items) {
        std::cout << "Error in Pricing MIP: could not find good reduced cost" << std::endl;
        return -1; // return standard pricing solution
    }

    solution.clear();

    if (highs->getModelStatus() == HighsModelStatus::kOptimal) {
        const auto& sol = highs->getSolution();

        for (int j = 0; j < _instance->items; ++j) {
            if (sol.col_value[j] > 0.5) {
                solution.push_back(j);
            }
        }
    }

    return highs->getObjectiveValue();
}


BppcTemplatePrice::BppcTemplatePrice(const BppcTemplatePrice& copy) {
    _rmp = copy._rmp;
    _instance = copy._instance;
    _template = copy._template;
    _mip = std::make_unique<PricingBlockVector<BppcPricingMIP>>(copy._machines);
    _mip->init(*_instance);
}

void BppcTemplatePrice::init(Highs* rmp, BppcInstance* instance) {
    _rmp = rmp;
    _instance = instance;
    _template.init(_machines, instance->items);
	_mip.reset(new PricingBlockVector<BppcPricingMIP>(_machines));
	_mip->init(*_instance);
}

double BppcTemplatePrice::optimize(const std::vector<double>& duals, PricingBlockVector<BppcPricing>& pricing, std::vector<double>& reduced_costs) {
	reduced_costs.resize(_machines, 0.0);

	// TODO: only need to solve dantzig subproblem once for optimal pricing, then template pricing can be used
	double tmp = pricing[0].optimize(duals, _farkas ? 0 : -1);
	if (tmp <= 1e-6) {
		reduced_costs.assign(_machines, -1);
		return tmp;
	}

	//for (int m = 0; m < _machines; ++m) {
	//	reduced_costs[m] = _mip->_pricing[m].optimize_template(_template._template_columns[m], duals, -1);
	//	pricing[m].solution.swap(_mip->_pricing[m].solution);
	//}

    highs::parallel::for_each(0, _machines, [&](HighsInt start, HighsInt end) {
        for (int m = start; m < end; ++m) {
            reduced_costs[m] = _mip->_pricing[m].optimize_template(_template._template_columns[m], duals, _farkas ? 0 : -1);
            pricing[m].solution.swap(_mip->_pricing[m].solution);
        }
    });

	// check for duplicates
	std::vector<std::vector<int>> item_seen;

	for (int m = 0; m < _machines; ++m) {
		if (reduced_costs[m] > 0) {
			// check if we have seen this item before
			for (auto& col : item_seen) {
				if (col == pricing[m].solution) {
					reduced_costs[m] = -1; // mark as duplicate
					break;
				}
			}

			if (reduced_costs[m] > 0) {
				item_seen.push_back(pricing[m].solution);
			}
		}
	}

    return tmp;
}

// Helper: Compute Hamming distance between two boolean vectors
static int hamming_distance(const std::vector<int>& a, const std::vector<double>& b) {
	int dist = 0;
	for (size_t i = 0; i < a.size(); ++i) {
		dist += (a[i] > 0) != (b[i] > 0);

		dist += 0.001 * ((a[i] < 0) != (b[i] < 0));
	}

	return dist;
}

// Helper: Compute the centroid (majority vote) of a cluster of integer vectors
static std::vector<double> compute_centroid(const std::vector<std::vector<int>>& cluster) {
	if (cluster.empty()) 
		return {};

	size_t dim = cluster[0].size();
	std::vector<int> count(dim, 0);

	for (const auto& vec : cluster) {
		for (size_t i = 0; i < dim; ++i)
			count[i] += vec[i];
	}

	std::vector<double> centroid(dim, 0.0);
	for (size_t i = 0; i < dim; ++i)
		centroid[i] = count[i] / double(cluster.size());

	return centroid;
}

void kmeans_cluster_columns(int K, std::vector<std::vector<int>>& columns, std::vector<std::vector<double>>& centroids) {
	if (columns.empty() || K <= 0) return;
	size_t N = columns.size();
	size_t dim = columns[0].size();

	// initialize K centroids
	int chunk_size = dim / K;

	for (int k = 0; k < K - 1; ++k) {
		std::fill(centroids[k].begin(), centroids[k].end(), 0.0);
		std::fill(centroids[k].begin() + k * chunk_size, centroids[k].begin() + (k + 1) * chunk_size, 1.0);
	}

	std::fill(centroids[K - 1].begin(), centroids[K - 1].end(), 0.0);
	std::fill(centroids[K - 1].begin() + (K - 1) * chunk_size, centroids[K - 1].end(), 1);

	std::vector<int> assignments(N, -1);
	bool changed = true;
	int max_iter = 1000;
	while (changed && max_iter--) {
		changed = false;

		// Assignment step
		for (size_t i = 0; i < N; ++i) {
			int best_k = 0;
			int best_dist = hamming_distance(columns[i], centroids[0]);

			for (int k = 1; k < K; ++k) {
				int dist = hamming_distance(columns[i], centroids[k]);
				if (dist < best_dist) {
					best_dist = dist;
					best_k = k;
				}
			}
			if (assignments[i] != best_k) {
				assignments[i] = best_k;
				changed = true;
			}
		}

		// Update step
		std::vector<std::vector<std::vector<int>>> groups(K);

		for (size_t i = 0; i < N; ++i)
			groups[assignments[i]].push_back(columns[i]);

		for (int k = 0; k < K; ++k) {
			if (!groups[k].empty())
				centroids[k] = compute_centroid(groups[k]);
		}
	}

	//// Print the assigned columns for debugging
	//auto tmp = sorted_index(K,
	//	[&](const int ai, const int bi) {
	//		const std::vector<double>& a = centroids[ai];
	//		const std::vector<double>& b = centroids[bi];

	//		double a_size = std::accumulate(a.begin(), a.end(), 0.0, [](double acc, double i) { return acc + int(i > 0); });
	//		double b_size = std::accumulate(b.begin(), b.end(), 0.0, [](double acc, double i) { return acc + int(i > 0); });

	//		double a_index = 0.0;
	//		double b_index = 0.0;

	//		for (int i = 0; i < a.size(); ++i) {
	//			if (a[i] > 0) {
	//				a_index += i;
	//			}
	//			if (b[i] > 0) {
	//				b_index += i;
	//			}
	//		}

	//		return a_index * b_size < b_index * a_size;
	//	});

	//for (int k = 0; k < K; ++k) {
	//	for (int n = 0; n < N; ++n) {
	//		if (assignments[n] == k) {
	//			std::vector<int> indices;
	//			for (int i = 0; i < dim; ++i) {
	//				if (columns[n][i] > 0) {
	//					indices.push_back(i);
	//				}
	//			}

	//			std::cout << k << ": " << join(indices, ",") << std::endl;
	//		}
	//	}
	//}

	// Sort the centroids by their average non-zero indices
	std::stable_sort(centroids.begin(), centroids.end(),
		[](const std::vector<double>& a, const std::vector<double>& b) {
			double a_size = std::accumulate(a.begin(), a.end(), 0.0, [](double acc, double i) { return acc + int(i > 0); });
			double b_size = std::accumulate(b.begin(), b.end(), 0.0, [](double acc, double i) { return acc + int(i > 0); });

			double a_index = 0.0;
			double b_index = 0.0;

			for (int i = 0; i < a.size(); ++i) {
				if (a[i] > 0) {
					a_index += i;
				}
				if (b[i] > 0) {
					b_index += i;
				}
			}

			return a_index * b_size < b_index * a_size;
		});

}

// Since the number of bins/machines is unknown (being optimized),
// We want to update the template by grouping the "machines" together in the RMP solution.
void BppcTemplatePrice::update() {
    _machines = std::floor(_rmp->getObjectiveValue() + 1e-6);

	// Choose top X columns with highest solution value, and group the other columns by their similarity to the top X columns.
	int num_vertices = _rmp->getNumRow();
	const auto& solution = _rmp->getSolution();
	const auto& lp = _rmp->getLp();

	std::vector<int> basis_columns;

	for (int i = solution.col_value.size() - 1; i >= 0; --i) {
		if (solution.col_value[i] > 1e-6) {
			basis_columns.push_back(i);
		}
	}

	// select top X columns (decreasing order)
	std::stable_sort(basis_columns.begin(), basis_columns.end(),
		[&](int a, int b) {
			double na_index = std::accumulate(lp.a_matrix_.index_.data() + lp.a_matrix_.start_[a],
				lp.a_matrix_.index_.data() + lp.a_matrix_.start_[a + 1],
				static_cast<uint64_t>(0)) / (lp.a_matrix_.start_[a + 1] - lp.a_matrix_.start_[a]);

			double nb_index = std::accumulate(lp.a_matrix_.index_.data() + lp.a_matrix_.start_[b],
				lp.a_matrix_.index_.data() + lp.a_matrix_.start_[b + 1],
				static_cast<uint64_t>(0)) / (lp.a_matrix_.start_[b + 1] - lp.a_matrix_.start_[b]);

			return na_index < nb_index;
		});

	if (basis_columns.size() > _machines) {
		_template._template_columns.resize(_machines, std::vector<double>(num_vertices, 0));

		std::vector<std::vector<int>> cols(basis_columns.size(), std::vector<int>(num_vertices, 0));

		for (int c = 0; c < basis_columns.size(); ++c) {
			auto col = basis_columns[c];
			auto start = lp.a_matrix_.index_.begin() + lp.a_matrix_.start_[col];
			auto end = lp.a_matrix_.index_.begin() + lp.a_matrix_.start_[col + 1];
			for (auto it = start; it != end; ++it) {
				cols[c][*it] = 1;

				//// set conflict columns from instance to -1
				//for (auto conflict : _instance->conflict_graph[*it]) {
				//	cols[c][conflict] = -1;
				//}
			}
		}

		kmeans_cluster_columns(_machines, cols, _template._template_columns);

		//// min sum_cdb {d_cd y_cdb}
		//// sum_b { y_ccb } == 1, c in cols
		//// sum_c { y_ccb } >= 1, b in bins
		//// -1 <= y_cdb + - y_c1c1b - y_c2c2b2 <= 1, c1, c2 in cols, b in bins

		//HighsModel compact_model;

		//int BINS = _machines;
		//int ITEMS = _instance->items;
		//int COLS = basis_columns.size();

		//auto Y_cdb = [&](int c, int d, int b) { return b * COLS * COLS + c * COLS + d; };

		//compact_model.lp_.num_col_ = BINS * COLS * COLS;
		//compact_model.lp_.num_row_ = COLS + BINS + COLS * COLS * BINS;
		//compact_model.lp_.offset_ = 0;
		//compact_model.lp_.sense_ = ObjSense::kMinimize;

		//// columns
		//compact_model.lp_.col_lower_.resize(compact_model.lp_.num_col_, 0);
		//compact_model.lp_.col_upper_.resize(compact_model.lp_.num_col_, 1);

		//// rows
		//compact_model.lp_.row_lower_.resize(compact_model.lp_.num_row_, 0);
		//compact_model.lp_.row_upper_.resize(compact_model.lp_.num_row_, kHighsInf);

		//// objective
		//compact_model.lp_.col_cost_.assign(compact_model.lp_.num_col_, 1);

		//// calculate distance between columns
		//for (int c1 = 0; c1 < COLS; ++c1) {
		//	auto col1 = basis_columns[c1];
		//	auto start1 = lp.a_matrix_.index_.begin() + lp.a_matrix_.start_[col1];
		//	auto end1 = lp.a_matrix_.index_.begin() + lp.a_matrix_.start_[col1 + 1];

		//	std::vector<bool> col1_items(ITEMS, false);
		//	for (auto it = start1; it != end1; ++it) {
		//		col1_items[*it] = true;
		//	}

		//	for (int c2 = 0; c2 < COLS; ++c2) {
		//		int distance = 0;

		//		if (c1 == c2) {
		//			// same column, distance is 0
		//			distance = 0;
		//		}
		//		else {
		//			auto col2 = basis_columns[c2];
		//			auto start2 = lp.a_matrix_.index_.begin() + lp.a_matrix_.start_[col2];
		//			auto end2 = lp.a_matrix_.index_.begin() + lp.a_matrix_.start_[col2 + 1];

		//			std::vector<bool> col2_items(ITEMS, false);
		//			for (auto it = start2; it != end2; ++it) {
		//				col2_items[*it] = true;
		//			}

		//			for (int i = 0; i < ITEMS; ++i) {
		//				if (col1_items[i] ^ col2_items[i]) {
		//					++distance;
		//				}
		//			}
		//		}

		//		for (int b = 0; b < BINS; ++b) {
		//			compact_model.lp_.col_cost_[Y_cdb(c1, c2, b)] = distance;
		//		}
		//	}
		//}


		//// initialize empty matrix
		//compact_model.lp_.a_matrix_.num_col_ = compact_model.lp_.num_col_;
		//compact_model.lp_.a_matrix_.num_row_ = compact_model.lp_.num_row_;
		//compact_model.lp_.a_matrix_.format_ = MatrixFormat::kRowwise;
		//compact_model.lp_.a_matrix_.start_ = { 0 };

		//int row = 0;

		//for (int c = 0; c < COLS; ++c) {
		//	for (int b = 0; b < BINS; ++b) {
		//		compact_model.lp_.a_matrix_.index_.push_back(Y_cdb(c, c, b));
		//		compact_model.lp_.a_matrix_.value_.push_back(1);
		//	}

		//	compact_model.lp_.a_matrix_.start_.push_back(compact_model.lp_.a_matrix_.index_.size());

		//	compact_model.lp_.row_lower_[row] = 1;
		//	compact_model.lp_.row_upper_[row] = 1;
		//	++row;
		//}

		//for (int b = 0; b < BINS; ++b) {
		//	for (int c = 0; c < COLS; ++c) {
		//		compact_model.lp_.a_matrix_.index_.push_back(Y_cdb(c, c, b));
		//		compact_model.lp_.a_matrix_.value_.push_back(1);
		//	}

		//	compact_model.lp_.a_matrix_.start_.push_back(compact_model.lp_.a_matrix_.index_.size());

		//	compact_model.lp_.row_lower_[row] = 1;
		//	compact_model.lp_.row_upper_[row] = COLS;
		//	++row;
		//}


		//for (int c1 = 0; c1 < COLS; ++c1) {
		//	for (int c2 = 0; c2 < COLS; ++c2) {
		//		for (int b = 0; b < BINS; ++b) {
		//			if (c1 == c2) {
		//				// skip same column
		//				compact_model.lp_.a_matrix_.start_.push_back(compact_model.lp_.a_matrix_.index_.size());
		//				compact_model.lp_.row_lower_[row] = 0;
		//				compact_model.lp_.row_upper_[row] = 0;
		//			}
		//			else {
		//				compact_model.lp_.a_matrix_.index_.push_back(Y_cdb(c1, c2, b));
		//				compact_model.lp_.a_matrix_.value_.push_back(1);

		//				compact_model.lp_.a_matrix_.index_.push_back(Y_cdb(c1, c1, b));
		//				compact_model.lp_.a_matrix_.value_.push_back(-1);

		//				compact_model.lp_.a_matrix_.index_.push_back(Y_cdb(c2, c2, b));
		//				compact_model.lp_.a_matrix_.value_.push_back(-1);

		//				compact_model.lp_.a_matrix_.start_.push_back(compact_model.lp_.a_matrix_.index_.size());
		//				compact_model.lp_.row_lower_[row] = -1;
		//				compact_model.lp_.row_upper_[row] = 1;
		//			}
		//			++row;
		//		}
		//	}
		//}

		//compact_model.lp_.integrality_.assign(compact_model.lp_.num_col_, HighsVarType::kInteger);
		//Highs highs;
		//highs.setOptionValue("time_limit", 20);
		//highs.setOptionValue("output_flag", false);
		//highs.passModel(compact_model);
		//highs.run();

		//auto sol = highs.getSolution();
		//std::vector<std::vector<std::vector<int>>> collection(_machines);

		//// print out the classification of the columns
		//for (int c = 0; c < COLS; ++c) {
		//	for (int b = 0; b < BINS; ++b) {
		//		if (sol.col_value[Y_cdb(c, c, b)] > 0) {
		//			auto col = basis_columns[c];
		//			auto start = lp.a_matrix_.index_.begin() + lp.a_matrix_.start_[col];
		//			auto end = lp.a_matrix_.index_.begin() + lp.a_matrix_.start_[col + 1];

		//			collection[b].push_back(std::vector<int>(start, end));
		//			//std::cout << b << ": " << join(collection[b].back(), ",") << std::endl;
		//		}
		//	}
		//}

		//for (int k = 0; k < _machines; ++k) {
		//	for (auto& colll : collection[k]) {
		//		std::cout << k << ": " << join(colll, ",") << std::endl;
		//	}
		//}

		//// normalize centers
		//std::vector<double> avg(_machines, 0.0);

		//for (int k = 0; k < _machines; ++k) {
		//	auto& center = _template._template_columns[k];
		//	int count = 0;

		//	for (auto& colll : collection[k]) {
		//		for (int i : colll) {
		//			++center[i];
		//			avg[k] += i;
		//			++count;
		//		}
		//	}
		//	if (count > 0) {
		//		avg[k] /= count;
		//	}
		//	else {
		//		avg[k] = 0.0;
		//	}

		//	int max = 0;
		//	for (int p = 0; p < num_vertices; ++p) {
		//		if (center[p] > max) {
		//			max = center[p];
		//		}
		//	}

		//	for (int p = 0; p < num_vertices; ++p) {
		//		if (max > 0) {
		//			center[p] /= max;
		//		}
		//		else {
		//			center[p] = 0.0;
		//		}
		//		center[p] = (center[p] > 1 - 1e-6) - (center[p] < 1e-6);
		//	}
		//}

		//auto tmp = sorted_index(_machines, [&](int a, int b) {
		//	return avg[a] < avg[b];
		//});

		//auto tmp2 = _template._template_columns;

		//for (int k = 0; k < _machines; ++k) {
		//	_template._template_columns[k] = tmp2[tmp[k]];
		//}

		for (auto& b : _template._template_columns) {
			for (auto& c : b) {
				c = (c > 1 - 0.7) - (c < 1e-6);
			}
		}
	}
	else {
		std::vector<std::vector<double>> centers(basis_columns.size(), std::vector<double>(num_vertices, 0.0));

		for (int k = 0; k < basis_columns.size(); ++k) {
			auto start = lp.a_matrix_.index_.begin() + lp.a_matrix_.start_[basis_columns[k]];
			auto end = lp.a_matrix_.index_.begin() + lp.a_matrix_.start_[basis_columns[k] + 1];

			//for (auto it = start; it != end; ++it) {
			//	centers[k][*it] = 1;
			//}

			for (auto it = start; it != end; ++it) {
				centers[k][*it] = 1;

				// set conflict columns from instance to -1
				for (auto conflict : _instance->conflict_graph[*it]) {
					centers[k][conflict] = -1;
				}
			}
		}

		_template._template_columns = centers;

		//for (auto& b : _template._template_columns) {
		//	for (auto& c : b) {
		//		c = (c > 1 - 1e-6) - (c < -1e-6);
		//	}
		//}
	}












//	// select top X columns (decreasing order)
//	std::stable_sort(basis_columns.begin(), basis_columns.end(),
//		[&](int a, int b) {
//			double na_index = std::accumulate(lp.a_matrix_.index_.data() + lp.a_matrix_.start_[a],
//				lp.a_matrix_.index_.data() + lp.a_matrix_.start_[a + 1],
//				static_cast<uint64_t>(0)) / (lp.a_matrix_.start_[a + 1] - lp.a_matrix_.start_[a]);
//
//			double nb_index = std::accumulate(lp.a_matrix_.index_.data() + lp.a_matrix_.start_[b],
//				lp.a_matrix_.index_.data() + lp.a_matrix_.start_[b + 1],
//				static_cast<uint64_t>(0)) / (lp.a_matrix_.start_[b + 1] - lp.a_matrix_.start_[b]);
//
//			return na_index < nb_index;
////			return solution.col_value[a] > solution.col_value[b] || solution.col_value[a] == solution.col_value[b] && na_index < nb_index;
//		});
//
//	if (basis_columns.size() > _machines) {
//		_template._template_columns.resize(_machines, std::vector<double>(num_vertices, 0));
//
//		int K = 1;
//
//		int fixed = 0;
//		std::vector<std::vector<std::vector<int>>> collection(_machines);
//		std::vector<int> counts(_machines, 0);
//
//		auto col = basis_columns[0];
//		auto start = lp.a_matrix_.index_.begin() + lp.a_matrix_.start_[col];
//		auto end = lp.a_matrix_.index_.begin() + lp.a_matrix_.start_[col + 1];
//		auto lambda = solution.col_value[col];
//
//		collection[0].push_back(std::vector<int>(start, end));
//		//std::cout << 0 << ": " << lambda << " - " << join(collection[0].back(), ",") << std::endl;
//
//		auto& center = _template._template_columns[0];
//		std::fill(center.begin(), center.end(), 0.0); // reset
//
//		for (auto it = start; it != end; ++it) {
//			center[*it] = 1;
//
//			//// set conflict columns from instance to -1
//			//for (auto conflict : _instance->conflict_graph[*it]) {
//			//	center[conflict] = -1;
//			//}
//		}
//
//		// Initialize centers with columns having highest solution value
//		for (int i = 1, size = basis_columns.size(); i < size; ++i) {
//			auto col = basis_columns[i];
//			auto start = lp.a_matrix_.index_.begin() + lp.a_matrix_.start_[col];
//			auto end = lp.a_matrix_.index_.begin() + lp.a_matrix_.start_[col + 1];
//			auto length = end - start;
//			auto lambda = solution.col_value[col];
//
//			double best_distance = 0; // maximize common entries
//			double best_conflict = std::numeric_limits<double>::max();  // minimize conflict differences
//			double dual_tie_break = std::numeric_limits<double>::max();
//			int best_cluster = -1;
//
//			// hack
//			std::vector<double> tmp(num_vertices, 0.0);
//			for (auto it = start; it != end; ++it) {
//				tmp[*it] = 1;
//
//				//// set conflict columns from instance to -1
//				//for (auto conflict : _instance->conflict_graph[*it]) {
//				//	tmp[conflict] = -1;
//				//}
//			}
//
//			for (int k = 0; k < K; k++) {
//				double distance = 0;
//				double conflict = 0;
//				double dual_distance = 0;
//				auto& center = _template._template_columns[k];
//
//				for (int p = 0; p < num_vertices; p++) {
//					distance += (double)(center[p] > 0 & tmp[p] > 0);  // maximize common entries
//					conflict += (double)((center[p] < 0 | tmp[p] < 0) & (center[p] < 0 ^ tmp[p] < 0));
//
//					dual_distance += (double)((center[p] < 0 & tmp[p] > 0) + (center[p] > 0 & tmp[p] < 0)); // minimize collisions!
//				}
//
//				if (best_distance < distance ||
//					best_distance == distance && dual_tie_break > dual_distance ||
//					dual_tie_break == dual_distance && best_distance == distance && best_conflict > conflict) {
//
//					best_distance = distance;
//					dual_tie_break = dual_distance;
//					best_conflict = conflict;
//					best_cluster = k;
//				}
//			}
//
//			if (K < _machines && 2 * best_distance + 1 <= length) {
//				collection[K].push_back(std::vector<int>(start, end));
//				//std::cout << K << ": " << lambda << " - " << join(collection[K].back(), ",") << std::endl;
//
//				auto& center = _template._template_columns[K];
//				std::fill(center.begin(), center.end(), 0.0); // reset
//
//				for (auto it = start; it != end; ++it) {
//					center[*it] = 1;
//
//					//// set conflict columns from instance to -1
//					//for (auto conflict : _instance->conflict_graph[*it]) {
//					//	center[conflict] = -1;
//					//}
//				}
//
//				++K;
//			}
//			else {
//				auto& center = _template._template_columns[best_cluster];
//				collection[best_cluster].push_back(std::vector<int>(start, end));
//				//std::cout << best_cluster << ": " << lambda << " - " << join(collection[best_cluster].back(), ",") << std::endl;
//
//				////for (auto it = start; it != end; ++it) {
//				////	center[*it] += lambda;
//				////}
//				for (int p = 0; p < num_vertices; p++) {
//					center[p] += tmp[p];
//				}
//			}
//		}
//
//		//// Initialize centers with columns having highest solution value
//		//for (int k = 0; k < _machines; ++k) {
//		//	auto col = basis_columns[k];
//		//	auto start = lp.a_matrix_.index_.begin() + lp.a_matrix_.start_[col];
//		//	auto end = lp.a_matrix_.index_.begin() + lp.a_matrix_.start_[col + 1];
//		//	auto lambda = solution.col_value[col];
//
//		//	if (lambda >= 1 - 1e-6) {
//		//		++fixed;
//		//	}
//
//		//	collection[k].push_back(std::vector<int>(start, end));
//		//	std::cout << k << ": " << lambda << " - " << join(collection[k].back(), ",") << std::endl;
//
//		//	auto& center = _template._template_columns[k];
//		//	std::fill(center.begin(), center.end(), 0.0); // reset
//
//		//	for (auto it = start; it != end; ++it) {
//		//		center[*it] = 1;
//
//		//		//// set conflict columns from instance to -1
//		//		//for (auto conflict : _instance->conflict_graph[*it]) {
//		//		//	center[conflict] = -1;
//		//		//}
//		//	}
//
//		//	//++counts[k];
//
//		//	//// hack
//		//	//std::vector<double> tmp(num_vertices, 0.0);
//		//	//for (auto it = start; it != end; ++it) {
//		//	//	tmp[*it] = 1;
//		//	//}
//
//		//	//for (int k_ = 0; k_ < k; k_++) {
//		//	//	double distance = 0;
//		//	//	auto& center = _template._template_columns[k];
//
//		//	//	for (int p = 0; p < num_vertices; p++) {
//		//	//		distance += (double)(center[p] > 0 & tmp[p] > 0);  // maximize common entries
//		//	//	}
//
//		//	//	if (distance > 0) {
//		//	//		for (int p = 0; p < num_vertices; p++) {
//		//	//			_template._template_columns[k][p] += tmp[p];
//		//	//		}
//
//		//	//		++counts[k];
//		//	//	}
//		//	//}
//		//}
//
//		//// Add remaining points to their nearest cluster
//		//for (int i = _machines, size = basis_columns.size(); i < size; ++i) {
//		//	double best_distance = 0; // maximize common entries
//		//	double best_conflict = std::numeric_limits<double>::max();  // minimize conflict differences
//		//	double dual_tie_break = std::numeric_limits<double>::max();
//		//	int best_cluster = -1;
//
//		//	auto col = basis_columns[i];
//		//	auto start = lp.a_matrix_.index_.begin() + lp.a_matrix_.start_[col];
//		//	auto end = lp.a_matrix_.index_.begin() + lp.a_matrix_.start_[col + 1];
//		//	auto lambda = solution.col_value[col];
//
//		//	// hack
//		//	std::vector<double> tmp(num_vertices, 0.0);
//		//	for (auto it = start; it != end; ++it) {
//		//		tmp[*it] = 1;
//
//		//		//// set conflict columns from instance to -1
//		//		//for (auto conflict : _instance->conflict_graph[*it]) {
//		//		//	tmp[conflict] = -1;
//		//		//}
//		//	}
//
//		//	for (int k = 0; k < _machines; k++) {
//		//		double distance = 0;
//		//		double conflict = 0;
//		//		double dual_distance = 0;
//		//		auto& center = _template._template_columns[k];
//
//		//		for (int p = 0; p < num_vertices; p++) {
//		//			distance += (double)(center[p] > 0 & tmp[p] > 0);  // maximize common entries
//		//			conflict += (double)((center[p] < 0 | tmp[p] < 0) & (center[p] < 0 ^ tmp[p] < 0));
//
//		//			dual_distance += (double)((center[p] < 0 & tmp[p] > 0) + (center[p] > 0 & tmp[p] < 0)); // minimize collisions!
//		//			//dual_distance += solution.row_dual[p] * (double)(center[p] > 0 | tmp[p] > 0);
//		//		}
//
//		//		//if (distance > 0) {
//		//		//	for (int p = 0; p < num_vertices; p++) {
//		//		//		_template._template_columns[k][p] += tmp[p];
//		//		//	}
//
//		//		//	++counts[k];
//		//		//}
//
//		//		//if (best_distance < distance || best_distance == distance && best_conflict > conflict) {
//		//		//if (dual_tie_break > dual_distance || 
//		//		//	dual_tie_break == dual_distance && best_distance < distance ||
//		//		//	dual_tie_break == dual_distance && best_distance == distance && best_conflict > conflict) {
//
//		//		if (best_distance < distance ||
//		//			best_distance == distance && dual_tie_break > dual_distance ||
//		//			dual_tie_break == dual_distance && best_distance == distance && best_conflict > conflict) {
//
//		//			best_distance = distance;
//		//			dual_tie_break = dual_distance;
//		//			best_conflict = conflict;
//		//			best_cluster = k;
//		//		}
//		//	}
//
//		//	auto& center = _template._template_columns[best_cluster];
//		//	collection[best_cluster].push_back(std::vector<int>(start, end));
//		//	std::cout << best_cluster << ": " << lambda << " - " << join(collection[best_cluster].back(), ",") << std::endl;
//
//
//		//	////for (auto it = start; it != end; ++it) {
//		//	////	center[*it] += lambda;
//		//	////}
//		//	for (int p = 0; p < num_vertices; p++) {
//		//		center[p] += tmp[p];
//		//	}
//		//}
//
//		//for (auto& b : _template._template_columns) {
//		//	for (auto& c : b) {
//		//		c = (c > 1 - 1e-6) - (c < -1e-6);
//		//	}
//		//}
//
//		// normalize centers
//		for (int k = 0; k < _machines; ++k) {
//			auto& center = _template._template_columns[k];
//
//			int max = 0;
//			for (int p = 0; p < num_vertices; ++p) {
//				if (center[p] > max) {
//					max = center[p];
//				}
//			}
//
//			for (int p = 0; p < num_vertices; ++p) {
//				if (max > 0) {
//					center[p] /= max;
//				}
//				else {
//					center[p] = 0.0;
//				}
//				center[p] = (center[p] > 1 - 1e-6) - (center[p] < 1e-6);
//			}
//		}
//	}
//	else {
//		std::vector<std::vector<double>> centers(basis_columns.size(), std::vector<double>(num_vertices, 0.0));
//
//		for (int k = 0; k < basis_columns.size(); ++k) {
//			auto start = lp.a_matrix_.index_.begin() + lp.a_matrix_.start_[basis_columns[k]];
//			auto end = lp.a_matrix_.index_.begin() + lp.a_matrix_.start_[basis_columns[k] + 1];
//
//			//for (auto it = start; it != end; ++it) {
//			//	centers[k][*it] = 1;
//			//}
//
//			for (auto it = start; it != end; ++it) {
//				centers[k][*it] = 1;
//
//				// set conflict columns from instance to -1
//				for (auto conflict : _instance->conflict_graph[*it]) {
//					centers[k][conflict] = -1;
//				}
//			}
//		}
//
//		_template._template_columns = centers;
//
//		//for (auto& b : _template._template_columns) {
//		//	for (auto& c : b) {
//		//		c = (c > 1 - 1e-6) - (c < -1e-6);
//		//	}
//		//}
//	}
}
