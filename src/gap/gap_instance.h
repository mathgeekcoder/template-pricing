#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include "nlohmann/json.hpp"
#include "utils.h"

using json = nlohmann::json;

// Load a GAP instance from file
// The file format is as follows:
//  <number of machines> <number of jobs>
//  <cost matrix>
//  <demand matrix>
//  <capacity vector>
struct GapInstance {
    int jobs, machines;

    std::string name;
    std::vector<int> capacity;
    std::vector<std::vector<int>> demands;
    std::vector<std::vector<double>> costs;

    GapInstance(std::string file = "") {
        machines = jobs = 0;  // initialize to 0 in case of loading error
        name = std::filesystem::path(file).filename().string();

        // check file extension (JSON or original)
        std::string ext = std::filesystem::path(file).extension().string();

        if (ext == ".json") {
            load_json(read_file(file));
        }
        else if (ext == ".gz") {
            load_json(read_gz_file(file));
        }
        else {
            load_original(file);
        }
    }

    void load_json(std::string json_content) {
        json j = json::parse(json_content);
        machines = j["nAgents"];
        jobs = j["nTasks"];
        capacity = j["capacity"].get<std::vector<int>>();
        costs = j["cost"].get<std::vector<std::vector<double>>>();
        demands = j["usage"].get<std::vector<std::vector<int>>>();
    }

    void load_original(std::string file) {
        std::ifstream in(file);
        if (in.is_open()) {
            in >> machines >> jobs;
            capacity.resize(machines);
            costs.resize(machines, std::vector<double>(jobs));
            demands.resize(machines, std::vector<int>(jobs));
            for (int m = 0; m < machines; m++) {
                for (int j = 0; j < jobs; j++)
                    in >> costs[m][j];
            }
            for (int m = 0; m < machines; m++) {
                for (int j = 0; j < jobs; j++)
                    in >> demands[m][j];
            }
            for (int m = 0; m < machines; m++)
                in >> capacity[m];
        }
	}

    void save_original(const std::string& filename) {
        std::ofstream out(filename);
        out << machines << " " << jobs << "\n";
        for (int m = 0; m < machines; m++) {
            for (int j = 0; j < jobs; j++)
                out << costs[m][j] << " ";
            out << "\n";
        }
        for (int m = 0; m < machines; m++) {
            for (int j = 0; j < jobs; j++)
                out << demands[m][j] << " ";
            out << "\n";
        }
        for (int m = 0; m < machines; m++)
            out << capacity[m] << " ";
        out << "\n";
        out.close();
	}
};


