#include "column_generation.h"

HighsModel SetPartitionRestrictedProblem(uint32_t partitions, uint32_t blocks, ObjSense sense) {
    HighsModel rmp;

    rmp.lp_.num_col_ = 0;
    rmp.lp_.num_row_ = partitions + blocks;
    rmp.lp_.offset_ = 0;
    rmp.lp_.sense_ = sense;

    // rows
    rmp.lp_.row_lower_.resize(rmp.lp_.num_row_, 1);
    rmp.lp_.row_upper_.resize(rmp.lp_.num_row_, 1);

    // initialize empty matrix
    rmp.lp_.a_matrix_.num_col_ = rmp.lp_.num_col_;
    rmp.lp_.a_matrix_.num_row_ = rmp.lp_.num_row_;
    rmp.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
    rmp.lp_.a_matrix_.start_ = { 0 };
    return rmp;
}

HighsModel SetCoverRestrictedProblem(uint32_t partitions, uint32_t blocks, ObjSense sense) {
    HighsModel rmp;

    rmp.lp_.num_col_ = 0;
    rmp.lp_.num_row_ = partitions + blocks;
    rmp.lp_.offset_ = 0;
    rmp.lp_.sense_ = sense;

    // rows
    rmp.lp_.row_lower_.resize(rmp.lp_.num_row_, 1);
    rmp.lp_.row_upper_.resize(rmp.lp_.num_row_, 1);

    std::fill(rmp.lp_.row_upper_.begin(), rmp.lp_.row_upper_.begin() + partitions, kHighsInf);

    // initialize empty matrix
    rmp.lp_.a_matrix_.num_col_ = rmp.lp_.num_col_;
    rmp.lp_.a_matrix_.num_row_ = rmp.lp_.num_row_;
    rmp.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
    rmp.lp_.a_matrix_.start_ = { 0 };
    return rmp;
}
