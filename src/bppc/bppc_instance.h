#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>

// Each instance file is organized as follows :
// 
// n	W
// 1	w	a1	a2	...	ak
// i	w	a1	a2	...	ak
// n	W	a1	a2	...	ak
// 
// where:
//   * n is the number of items
//   * W is the bin capacity
//   * i is a number identifying the item
//   * w is the item weight
//   * a1, ..., ak are the items that are incompatible with the item i
struct BppcInstance {
	int items, capacity;
	std::string name;
	std::vector<int> weights;
	std::vector<std::vector<int32_t>> conflict_graph;

	BppcInstance(std::string file = "") {
		capacity = 0;  // initialize to 0 in case of loading error
        name = std::filesystem::path(file).filename().string();

		std::ifstream in(file);

		if (in.is_open() == false)
			throw std::runtime_error("File not found");

		in >> items >> capacity;

		std::string line;
		getline(in, line);

		weights.resize(items);
		conflict_graph.resize(items);

		// index weight [list of incompatible items]
		int item;

		for (int i = 0; i < items && getline(in, line); ++i) {
			std::stringstream ls(line);
			ls >> item >> weights[i];

			while (ls >> item) {
				conflict_graph[i].push_back(item - 1);
			}
		}
    }

	bool has_conflict(int i, int j) const {
		if (i == j) return false;  // no conflict with itself
		return std::binary_search(conflict_graph[i].begin(), conflict_graph[i].end(), j);
	}
};