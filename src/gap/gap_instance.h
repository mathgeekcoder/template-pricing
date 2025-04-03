#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>

// Load a GAP instance from file
// The file format is as follows:
//  <number of machines> <number of jobs>
//  <profit matrix>
//  <demand matrix>
//  <capacity vector>
struct GapInstance {
    int jobs, machines;

    std::string name;
    std::vector<int> capacity;
    std::vector<std::vector<int>> demands;
    std::vector<std::vector<double>> profit;

    GapInstance(std::string file = "") {
        machines = jobs = 0;  // initialize to 0 in case of loading error
        name = std::filesystem::path(file).filename().string();

        std::ifstream in(file);

        if (in.is_open()) {
            in >> machines >> jobs;

            capacity.resize(machines);
            profit.resize(machines, std::vector<double>(jobs));
            demands.resize(machines, std::vector<int>(jobs));

            for (int m = 0; m < machines; m++) {
                for (int j = 0; j < jobs; j++)
                    in >> profit[m][j];
            }

            for (int m = 0; m < machines; m++) {
                for (int j = 0; j < jobs; j++)
                    in >> demands[m][j];
            }

            for (int m = 0; m < machines; m++)
                in >> capacity[m];
        }
    }
};


