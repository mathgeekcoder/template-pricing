#include "utils.h"

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#define NOMINMAX

#include <windows.h>
#include <processthreadsapi.h>
#include <psapi.h>

void MemoryUsage::snapshot() {
    PROCESS_MEMORY_COUNTERS pmc;

    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        peak = pmc.PeakWorkingSetSize;
    }
}

size_t MemoryUsage::PeakSize() {
    PROCESS_MEMORY_COUNTERS pmc;

    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.PeakWorkingSetSize - peak;
    }

    return -1;
}

#endif

// Converts dense system of equations into sparse highs model
HighsModel construct_system(const std::vector<std::vector<double>>& constraints, const std::vector<std::tuple<double, double>>& bounds) {
    HighsModel model;
    model.lp_.num_col_ = constraints[0].size();
    model.lp_.num_row_ = constraints.size();
    model.lp_.sense_ = ObjSense::kMaximize;

    model.lp_.col_lower_.assign(model.lp_.num_col_, 0);
    model.lp_.col_upper_.assign(model.lp_.num_col_, 1);
    model.lp_.integrality_.assign(model.lp_.num_col_, HighsVarType::kInteger);
    model.lp_.row_lower_.resize(model.lp_.num_row_);
    model.lp_.row_upper_.resize(model.lp_.num_row_);

    for (int r = 0; r < model.lp_.num_row_; ++r) {
        model.lp_.row_lower_[r] = std::get<0>(bounds[r]);
        model.lp_.row_upper_[r] = std::get<1>(bounds[r]);
    }

    model.lp_.a_matrix_.num_col_ = model.lp_.num_col_;
    model.lp_.a_matrix_.num_row_ = model.lp_.num_row_;
    model.lp_.a_matrix_.format_ = MatrixFormat::kRowwise;

    int count = 0;

    for (int r = 0; r < model.lp_.num_row_; ++r) {
        for (int c = 0; c < model.lp_.num_col_; ++c) {
            if (constraints[r][c] != 0) {
                model.lp_.a_matrix_.index_.push_back(c);
                model.lp_.a_matrix_.value_.push_back(constraints[r][c]);
                ++count;
            }
        }

        model.lp_.a_matrix_.start_.push_back(count);
    }

    return model;
}

// Converts dense system of equations into sparse highs model
HighsModel construct_system(const std::vector<double>& constraint, const std::tuple<double, double> bounds){
    HighsModel model;
    model.lp_.num_col_ = constraint.size();
    model.lp_.num_row_ = 1;
    model.lp_.sense_ = ObjSense::kMaximize;

    model.lp_.col_lower_.assign(model.lp_.num_col_, 0);
    model.lp_.col_upper_.assign(model.lp_.num_col_, 1);
    model.lp_.integrality_.assign(model.lp_.num_col_, HighsVarType::kInteger);
    model.lp_.row_lower_.resize(model.lp_.num_row_);
    model.lp_.row_upper_.resize(model.lp_.num_row_);

    model.lp_.row_lower_[0] = std::get<0>(bounds);
    model.lp_.row_upper_[0] = std::get<1>(bounds);

    model.lp_.a_matrix_.num_col_ = model.lp_.num_col_;
    model.lp_.a_matrix_.num_row_ = model.lp_.num_row_;
    model.lp_.a_matrix_.format_ = MatrixFormat::kRowwise;

    int count = 0;

    for (int c = 0; c < model.lp_.num_col_; ++c) {
        if (constraint[c] != 0) {
            model.lp_.a_matrix_.index_.push_back(c);
            model.lp_.a_matrix_.value_.push_back(constraint[c]);
            ++count;
        }
    }

    model.lp_.a_matrix_.start_.push_back(count);

    return model;
}
