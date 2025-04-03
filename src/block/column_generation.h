#pragma once
#include <vector>
#include "utils.h"

// PricingBlockArray has a fixed number of blocks, and each block has a different pricing problem
template <typename... Pricing>
class PricingBlockArray { };

// PricingBlockVector has a variable number of blocks, and each block has the same pricing problem
template <typename Pricing>
struct PricingBlockVector {
	PricingBlockVector(int num_blocks) : _pricing(num_blocks) { }
	std::vector<Pricing> _pricing;

	Pricing& operator[](int i) {
		return _pricing[i];
	}

	const Pricing& operator[](int i) const {
		return _pricing[i];
	}

	size_t size() const {
		return _pricing.size();
	}

	template <typename... Args>
	void init(Args&&... args) {
#ifdef _DEBUG
		int k = 0;

		for (auto& p : _pricing) {
			p.init(k++, std::forward<Args>(args)...);
		}
#else
		highs::parallel::for_each(0, size(), [&](HighsInt start, HighsInt end) {
			for (int k = start; k < end; ++k) {
				_pricing[k].init(k, std::forward<Args>(args)...);
			}
		});
#endif
	}
};


// Standard Restricted Problems
HighsModel SetPartitionRestrictedProblem(uint32_t partitions, uint32_t blocks, ObjSense sense);
HighsModel SetCoverRestrictedProblem(uint32_t partitions, uint32_t blocks, ObjSense sense);

struct TemplatePricing {
	int _blocks = 0, _partitions = 0;
	std::vector<std::vector<double>> _template_columns;

	const std::vector<double>& operator[](int i) const {
		return _template_columns[i];
	}

	const size_t size() const {
		return _template_columns.size();
	}

	void init(const std::vector<std::vector<double>>& template_columns) {
		_template_columns = template_columns;
		_blocks = template_columns.size();
		_partitions = template_columns[0].size();// -1;
	}

	void init(int blocks, int partitions) {
		_template_columns.resize(blocks, std::vector<double>(partitions, 0.0));
		_blocks = blocks;
		_partitions = partitions;
	}

	void init(const std::vector<std::vector<HighsInt>>& template_columns, int partitions) {
		_template_columns.resize(template_columns.size(), std::vector<double>(partitions, 0));
		_blocks = template_columns.size();
		_partitions = partitions;

		for (int block = template_columns.size() - 1; block >= 0; --block) {
			for (int index : template_columns[block]) {
				_template_columns[block][index] = 1;
			}
		}

		for (auto& b : _template_columns) {
			for (auto& c : b) {
				c = (c > 1 - 1e-6) - (c < 1e-6);
			}
		}
	}

	template <typename HighsSolver>
	void update(const HighsSolution& solution, const HighsSolver& highs) {
		for (int block = 0; block < _blocks; ++block) {
			std::fill(_template_columns[block].begin(), _template_columns[block].end(), 0.0);
		}

		// assumes lp colwise
		for (int i = solution.col_value.size() - 1; i >= 0; --i) {
			if (solution.col_value[i] > 1e-6) {
				auto start = col_begin(highs, i);
				auto end = col_end(highs, i);

				auto block_index = std::max_element(start, end);
				auto& template_block = _template_columns[*block_index - _partitions];

				for (auto it = start; it != end; ++it) {
					if (it != block_index) {
						template_block[*it] += solution.col_value[i];
					}
				}
			}
		}

		for (auto& b : _template_columns) {
			for (auto& c : b) {
				c = (c > 1 - 1e-6) - (c < 1e-6);
			}
		}
	}
};

