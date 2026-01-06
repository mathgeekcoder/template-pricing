#pragma once
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include "utils.h"

#ifdef SUPPORT_GUROBI
#include "gurobi_c.h"
#include "highs/Highs.h"

// Provides a basic HiGHS API wrapper using the Gurobi C API
class GurobiHighs {
	inline static GRBenv* env = nullptr;

	GRBmodel* _model = nullptr;
	double _offset = 0.0;

	int _numvars = 0;
	int _numcons = 0;
	bool stop_requested = false;

	std::vector<int> basic_vars;
	HighsSolution solution;

public:
	GurobiHighs() {
		_offset = 0;
		start();
	}

	~GurobiHighs() {
		if (_model) {
			GRBfreemodel(_model);
			_model = nullptr;
		}
		// Do not free env here; keep it for lifetime similar to previous behavior.
	}

	void ensureModel() {
		if (!_model && env) {
			GRBnewmodel(env, &_model, "model", 0, nullptr, nullptr, nullptr, nullptr, nullptr);
		}
	}

	static void start() {
		if (env == nullptr) {
			GRBemptyenv(&env);
			if (env) {
				// set env to have no output by default
				GRBsetintparam(env, GRB_INT_PAR_OUTPUTFLAG, 0);
				GRBstartenv(env);
			}
		}
	}

	static void stop() {
		if (env) {
			GRBfreeenv(env);
			env = nullptr;
		}
	}

	HighsStatus run() {
		GRBsetcallbackfunc(_model, my_callback, this);
		GRBoptimize(_model);

		// get primal/dual solution
		int status = GRB_LOADED;
		GRBgetintattr(_model, GRB_INT_ATTR_STATUS, &status);

		if (status == GRB_OPTIMAL) {
			solution.col_value.resize(_numvars);
			GRBgetdblattrarray(_model, GRB_DBL_ATTR_X, 0, _numvars, solution.col_value.data());
			solution.value_valid = true;

			solution.row_dual.resize(_numcons);
			GRBgetdblattrarray(_model, GRB_DBL_ATTR_PI, 0, _numcons, solution.row_dual.data());
			solution.dual_valid = true;

			// I don't think this works if the constraint is equality (i.e., no slack)
			//// get row value from slack, i.e., value +- slack = rhs
			solution.row_value.resize(_numcons);
			GRBgetdblattrarray(_model, GRB_DBL_ATTR_SLACK, 0, _numcons, solution.row_value.data());

			for (int i = 0; i < _numcons; ++i) {
				double rhs = 0.0;
				GRBgetdblattrelement(_model, GRB_DBL_ATTR_RHS, i, &rhs);
				solution.row_value[i] = rhs - solution.row_value[i];
			}
		}

		return (status == GRB_OPTIMAL || status == GRB_INFEASIBLE) ? HighsStatus::kOk : HighsStatus::kError;
	}

	// Callback function for Gurobi
	static int __stdcall my_callback(GRBmodel* model, void* cbdata, int where, void* usrdata) {
		(void)cbdata; (void)where; // unused parameters

		GurobiHighs* gh = static_cast<GurobiHighs*>(usrdata);
		if (gh->stop_requested) {
			GRBterminate(model);
		}
		return 0; // 0 means continue unless terminated
	}

	void requestStop() {
		stop_requested = true;
	}

	HighsSolution& getSolution() { return solution; }

	int getNumCol() { return _numvars; }
	int getNumRow() { return _numcons; }

	double getColCost(int col) {
		double val = 0.0;
		GRBgetdblattrelement(_model, GRB_DBL_ATTR_OBJ, col, &val);
		return val;
	}

	bool getColsCost(int from, int to, double* costs) const {
		if (from > to || from >= _numvars) return false;
		to = std::min(to, _numvars - 1);
		GRBupdatemodel(_model);  // needed if new cols have been added, but haven't solved yet
		GRBgetdblattrarray(_model, GRB_DBL_ATTR_OBJ, from, to - from + 1, costs);
		return true;
	}

	double getObjectiveValue() {
		double val = 0.0;
		GRBgetdblattr(_model, GRB_DBL_ATTR_OBJVAL, &val);
		return val + _offset;
	}

	void changeObjectiveOffset(double offset) {
		_offset = offset;
	}

	void changeColsCost(int from, int to, const double* costs) {
		if (from > to || from >= _numvars) return;
		to = std::min(to, _numvars - 1);
		GRBsetdblattrarray(_model, GRB_DBL_ATTR_OBJ, from, to - from + 1, const_cast<double*>(costs));
	}

	void changeColBounds(HighsInt col, double lower, double upper) {
		GRBsetdblattrelement(_model, GRB_DBL_ATTR_LB, col, lower);
		GRBsetdblattrelement(_model, GRB_DBL_ATTR_UB, col, upper);
	}

	void changeRowBounds(HighsInt row, double lower, double upper) {
		if (!_model) return;
		// Try to update sense and RHS for simple cases
		if (lower == upper) {
			GRBsetcharattrelement(_model, GRB_CHAR_ATTR_SENSE, row, GRB_EQUAL);
			GRBsetdblattrelement(_model, GRB_DBL_ATTR_RHS, row, upper);
		}
		else if (lower <= -kHighsInf && upper < kHighsInf) {
			GRBsetcharattrelement(_model, GRB_CHAR_ATTR_SENSE, row, GRB_LESS_EQUAL);
			GRBsetdblattrelement(_model, GRB_DBL_ATTR_RHS, row, upper);
		}
		else if (upper >= kHighsInf && lower > -kHighsInf) {
			GRBsetcharattrelement(_model, GRB_CHAR_ATTR_SENSE, row, GRB_GREATER_EQUAL);
			GRBsetdblattrelement(_model, GRB_DBL_ATTR_RHS, row, lower);
		}
		else {
			// General range: no direct single-attribute mapping in C API.
			// Recreate the row: get coefficients, delete the old constraint and add a range constraint.
			// Fallback: set as <= upper
			GRBsetcharattrelement(_model, GRB_CHAR_ATTR_SENSE, row, GRB_LESS_EQUAL);
			GRBsetdblattrelement(_model, GRB_DBL_ATTR_RHS, row, upper);
		}
	}

	void changeCoeff(HighsInt row, HighsInt col, double value) {
		int r = static_cast<int>(row);
		int c = static_cast<int>(col);
		GRBchgcoeffs(_model, 1, &r, &c, &value);
	}

	void setOptionValue(std::string option, const char* value) {
		setOptionValue(option, std::string(value));
	}

	void setOptionValue(std::string option, std::string value) {
		if (option == "output_flag") {
			GRBsetintparam(env, GRB_INT_PAR_OUTPUTFLAG, value == "false" ? 0 : 1);
		}
		else if (option == "simplex_strategy") {
			int choice = GRB_METHOD_PRIMAL;
			if (value == "1" || value == "2" || value == "3")
				choice = GRB_METHOD_DUAL;

			GRBsetintparam(env, GRB_INT_PAR_METHOD, choice);
		}
		else if (option == kPresolveString) {
			GRBsetintparam(env, GRB_INT_PAR_PRESOLVE, value == "off" ? 0 : -1);
		}
		else if (option == kSolverString) {
			if (value == kIpmString)
				GRBsetintparam(env, GRB_INT_PAR_METHOD, GRB_METHOD_BARRIER);
		}
		else if (option == kRunCrossoverString) {
			GRBsetintparam(env, GRB_INT_PAR_CROSSOVER, value == "off" ? 0 : 1);
		}
		else if (option == "allow_unbounded_or_infeasible") {
			GRBsetintparam(env, GRB_INT_PAR_INFUNBDINFO, value == "true" ? 1 : 0);
		}
	}

	void setOptionValue(std::string option, bool value) {
		if (option == "output_flag") {
			GRBsetintparam(env, GRB_INT_PAR_OUTPUTFLAG, value ? 1 : 0);
		}
		else if (option == "allow_unbounded_or_infeasible") {
			GRBsetintparam(env, GRB_INT_PAR_INFUNBDINFO, value ? 1 : 0);
		}
	}

	void setOptionValue(std::string option, int value) {
		if (option == "threads") {
			GRBsetintparam(env, GRB_INT_PAR_THREADS, value);
		}
	}

	void setOptionValue(std::string option, double value) {
		if (option == "mip_abs_gap") {
			GRBsetdblparam(env, GRB_DBL_PAR_MIPGAPABS, value);
		}
		else if (option == "mip_rel_gap") {
			GRBsetdblparam(env, GRB_DBL_PAR_MIPGAP, value);
		}
	}

	// construct model from HiGHS model
	void passModel(HighsModel& model) {
		if (_model) {
			GRBfreemodel(_model);
			_model = nullptr;
		}

		// create empty model
		GRBnewmodel(env, &_model, "model", 0, nullptr, nullptr, nullptr, nullptr, nullptr);

		_numvars = model.lp_.num_col_;
		_numcons = model.lp_.num_row_;

		// set model sense
		GRBsetintattr(_model, GRB_INT_ATTR_MODELSENSE, (model.lp_.sense_ == ObjSense::kMinimize) ? GRB_MINIMIZE : GRB_MAXIMIZE);

		// add variables (no structural coefficients provided here)
		if (_numvars > 0) {
			std::vector<char> vtype(_numvars, GRB_CONTINUOUS);
			if (!model.lp_.integrality_.empty()) {
				for (int i = 0; i < _numvars; ++i) {
					if (model.lp_.integrality_[i] == HighsVarType::kInteger) {
						if (model.lp_.col_lower_[i] == 0 && model.lp_.col_upper_[i] == 1)
							vtype[i] = GRB_BINARY;
						else
							vtype[i] = GRB_INTEGER;
					}
				}
			}

			GRBaddvars(_model, _numvars, 0, nullptr, nullptr, nullptr,
				const_cast<double*>(model.lp_.col_cost_.data()),
				const_cast<double*>(model.lp_.col_lower_.data()),
				const_cast<double*>(model.lp_.col_upper_.data()),
				vtype.data(), nullptr);
		}

		// add constraints row by row
		HighsSparseMatrix m = model.lp_.a_matrix_;
		m.ensureRowwise();

		for (int row = 0; row < _numcons; ++row) {
			int start = m.start_[row];
			int end = m.start_[row + 1];
			int n = end - start;
			if (n == 0) {
				// empty row: add a trivial constraint depending on bounds
				if (model.lp_.row_lower_[row] == model.lp_.row_upper_[row]) {
					GRBaddconstr(_model, 0, nullptr, nullptr, GRB_EQUAL, model.lp_.row_upper_[row], nullptr);
				}
				else if (model.lp_.row_lower_[row] <= -kHighsInf && model.lp_.row_upper_[row] < kHighsInf) {
					GRBaddconstr(_model, 0, nullptr, nullptr, GRB_LESS_EQUAL, model.lp_.row_upper_[row], nullptr);
				}
				else if (model.lp_.row_upper_[row] >= kHighsInf && model.lp_.row_lower_[row] > -kHighsInf) {
					GRBaddconstr(_model, 0, nullptr, nullptr, GRB_GREATER_EQUAL, model.lp_.row_lower_[row], nullptr);
				}
				else {
					// free row: create a trivial <= large rhs
					GRBaddconstr(_model, 0, nullptr, nullptr, GRB_LESS_EQUAL, GRB_INFINITY, nullptr);
				}
				continue;
			}

			// build index and value arrays for this row
			int* inds = m.index_.data() + start;
			double* vals = m.value_.data() + start;

			if (model.lp_.row_lower_[row] == model.lp_.row_upper_[row]) {
				GRBaddconstr(_model, n, inds, vals, GRB_EQUAL, model.lp_.row_upper_[row], nullptr);
			}
			else if (model.lp_.row_lower_[row] <= -kHighsInf) {
				GRBaddconstr(_model, n, inds, vals, GRB_LESS_EQUAL, model.lp_.row_upper_[row], nullptr);
			}
			else if (model.lp_.row_upper_[row] >= kHighsInf) {
				GRBaddconstr(_model, n, inds, vals, GRB_GREATER_EQUAL, model.lp_.row_lower_[row], nullptr);
			}
			else {
				GRBaddrangeconstr(_model, n, inds, vals, model.lp_.row_lower_[row], model.lp_.row_upper_[row], nullptr);
			}
		}
	}

	HighsStatus addCols(int cols, double* costs, double* lower, double* upper, int nnz, int* starts, int* index, double* value) {
		ensureModel();
		for (int col = 0; col < cols; ++col) {
			int start = starts[col];
			int end = starts[col + 1];
			int n = end - start;
			if (n > 0) {
				GRBaddvar(_model, n, index + start, value + start, costs[col], lower[col], upper[col], GRB_CONTINUOUS, nullptr);
			}
			else {
				GRBaddvar(_model, 0, nullptr, nullptr, costs[col], lower[col], upper[col], GRB_CONTINUOUS, nullptr);
			}
			++_numvars;
		}

		return HighsStatus::kOk;
	}

	HighsStatus addCol(double cost, double lower, double upper, int nnz, int* index, double* value) {
		ensureModel();
		if (nnz > 0) {
			GRBaddvar(_model, nnz, index, value, cost, lower, upper, GRB_CONTINUOUS, nullptr);
		}
		else {
			GRBaddvar(_model, 0, nullptr, nullptr, cost, lower, upper, GRB_CONTINUOUS, nullptr);
		}
		++_numvars;

		return HighsStatus::kOk;
	}

	void deleteCols(int cols, int* index) {
		GRBdelvars(_model, cols, index);
		GRBupdatemodel(_model);
		GRBgetintattr(_model, GRB_INT_ATTR_NUMVARS, &_numvars);
	}

	void addRows(int rows, double* lower, double* upper, int nnz, int* starts, int* index, double* value) {
		int offset = 0;
		for (int r = 0; r < rows; ++r) {
			int start = starts[r];
			int end = starts[r + 1];
			int n = end - start;
			if (n > 0) {
				if (lower[r] == upper[r]) {
					GRBaddconstr(_model, n, index + start, value + start, GRB_EQUAL, upper[r], nullptr);
				}
				else if (lower[r] <= -kHighsInf) {
					GRBaddconstr(_model, n, index + start, value + start, GRB_LESS_EQUAL, upper[r], nullptr);
				}
				else if (upper[r] >= kHighsInf) {
					GRBaddconstr(_model, n, index + start, value + start, GRB_GREATER_EQUAL, lower[r], nullptr);
				}
				else {
					GRBaddrangeconstr(_model, n, index + start, value + start, lower[r], upper[r], nullptr);
				}
			}
			else {
				// empty row
				if (lower[r] == upper[r]) {
					GRBaddconstr(_model, 0, nullptr, nullptr, GRB_EQUAL, upper[r], nullptr);
				}
				else if (lower[r] <= -kHighsInf) {
					GRBaddconstr(_model, 0, nullptr, nullptr, GRB_LESS_EQUAL, upper[r], nullptr);
				}
				else if (upper[r] >= kHighsInf) {
					GRBaddconstr(_model, 0, nullptr, nullptr, GRB_GREATER_EQUAL, lower[r], nullptr);
				}
				else {
					GRBaddrangeconstr(_model, 0, nullptr, nullptr, lower[r], upper[r], nullptr);
				}
			}
			++offset;
			++_numcons;
		}
	}

	void deleteRows(HighsInt from, HighsInt to) {
		int len = static_cast<int>(to - from + 1);

		std::vector<int> idx(len);
		std::iota(idx.begin(), idx.end(), static_cast<int>(from));

		GRBdelconstrs(_model, len, idx.data());
		GRBupdatemodel(_model);
		GRBgetintattr(_model, GRB_INT_ATTR_NUMCONSTRS, &_numcons);
	}

	HighsModelStatus getModelStatus() {
		if (!_model) return HighsModelStatus::kNotset;
		int s = 0;
		GRBgetintattr(_model, GRB_INT_ATTR_STATUS, &s);
		switch (s) {
		case GRB_OPTIMAL:
			return HighsModelStatus::kOptimal;
		case GRB_INFEASIBLE:
			return HighsModelStatus::kInfeasible;
		case GRB_INF_OR_UNBD:
			return HighsModelStatus::kUnboundedOrInfeasible;
		default:
			return HighsModelStatus::kNotset;
		}
	}

	const HighsInt* getBasicVariablesArray() {
		basic_vars.clear();
		if (!_model) return basic_vars.data();
		basic_vars.resize(_numcons);
		GRBgetBasisHead(_model, basic_vars.data());
		return basic_vars.data();
	}

	void getDualRay(bool& has_dual_ray, double* dual_ray) {
		has_dual_ray = false;
		if (!_model) return;
		int status = 0;
		GRBgetintattr(_model, GRB_INT_ATTR_STATUS, &status);
		if (_numcons > 0 && (status == GRB_UNBOUNDED || status == GRB_INFEASIBLE)) {
			std::vector<double> farkas(_numcons, 0.0);
			GRBgetdblattrarray(_model, GRB_DBL_ATTR_FARKASDUAL, 0, _numcons, farkas.data());
			for (int c = 0; c < _numcons; ++c) dual_ray[c] = -farkas[c];
			has_dual_ray = true;
		}
	}

	HighsInfo getInfo() {
		HighsInfo info;
		if (!_model) return info;
		int barIterCount = 0;
		double iterCount = 0.0;
		GRBgetintattr(_model, GRB_INT_ATTR_BARITERCOUNT, &barIterCount);
		GRBgetdblattr(_model, GRB_DBL_ATTR_ITERCOUNT, &iterCount);
		info.ipm_iteration_count = barIterCount;
		info.simplex_iteration_count = iterCount;
		return info;
	}

	// get all non-zero row indices for this column
	std::vector<int> get_col(int col) const {
		std::vector<int> idx;
		if (!_model) return idx;

		int nnz = 0;
		GRBgetvars(_model, &nnz, nullptr, nullptr, nullptr, col, 1);
		if (nnz > 0) {
			idx.resize(nnz);
			std::vector<int> vbeg(1);
			std::vector<double> vals(nnz);
			GRBgetvars(_model, &nnz, vbeg.data(), idx.data(), vals.data(), col, 1);
		}
		return idx;
	}
};

#endif