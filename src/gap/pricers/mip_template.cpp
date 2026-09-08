#include "mip_template.h"

template <typename RmpSolver>
void GapPricingMIP<RmpSolver>::init(uint32_t index, const GapInstance& instance) {
    GapPricing::init(index, instance);

    // max tx
    //   0 <= wx <= C
    // eps <= dx <= inf
    //  lb <= tx <= ub
    HighsModel model;
    model.lp_.num_col_ = _instance->jobs;
    model.lp_.num_row_ = 3;
    model.lp_.a_matrix_.num_col_ = model.lp_.num_col_;
    model.lp_.a_matrix_.num_row_ = model.lp_.num_row_;

    model.lp_.offset_ = 0;
    model.lp_.sense_ = ObjSense::kMaximize;

    model.lp_.col_cost_.assign(_instance->jobs, 0);
    model.lp_.col_lower_.assign(_instance->jobs, 0);
    model.lp_.col_upper_.assign(_instance->jobs, 1);
    model.lp_.integrality_.assign(_instance->jobs, HighsVarType::kInteger);

    model.lp_.row_lower_ = { -kHighsInf, 0, 0 };
    model.lp_.row_upper_ = { (double)_instance->capacity[_machine], kHighsInf, kHighsInf };

    model.lp_.a_matrix_.format_ = MatrixFormat::kRowwise;
    model.lp_.a_matrix_.start_ = { 0, _instance->jobs, 2 * _instance->jobs, 3 * _instance->jobs };
    model.lp_.a_matrix_.index_.resize(3 * _instance->jobs, 0);
    model.lp_.a_matrix_.value_.resize(3 * _instance->jobs, 0);

    for (int i = 0; i < _instance->jobs; ++i) {
        model.lp_.a_matrix_.index_[i] = i;
        model.lp_.a_matrix_.index_[i + _instance->jobs] = i;
        model.lp_.a_matrix_.index_[i + 2 * _instance->jobs] = i;

        model.lp_.a_matrix_.value_[i] = _instance->demands[_machine][i];
    }

    highs.reset(new RmpSolver);
    highs->setOptionValue("output_flag", false);
    highs->setOptionValue("threads", 1);
    highs->passModel(model);
}

template <typename RmpSolver>
double GapPricingMIP<RmpSolver>::optimize_template(const std::vector<double>& template_obj, const std::vector<double>& duals, double offset) {
    // heuristic dual & template
    double tmp = optimize(duals, offset);
    if (tmp <= 1e-6) {
        return tmp;
    }

    highs->changeObjectiveOffset(0);
    highs->changeColsCost(0, _instance->jobs - 1, template_obj.data());
	highs->changeRowBounds(1, -offset, kHighsInf);
    highs->changeRowBounds(2, -_instance->jobs, kHighsInf);

    for (int j = 0; j < _instance->jobs; ++j)
        highs->changeCoeff(1, j, duals[j]);

    for (int j = 0; j < _instance->jobs; ++j)
        highs->changeCoeff(2, j, template_obj[j]);

    highs->run();
    double ht = highs->getObjectiveValue(), hd = offset;

    // switch objective, add new constraint
    highs->changeColsCost(0, _instance->jobs - 1, duals.data());
    highs->changeRowBounds(2, std::ceil(ht - 0.5), kHighsInf);
    highs->changeObjectiveOffset(offset);
    HighsStatus status = highs->run();

	// ensure good reduced costs, prevent infinite loop
	while (ht > -_instance->jobs && highs->getObjectiveValue() <= 1e-6) {
        --ht;
        highs->changeRowBounds(2, std::ceil(ht - 0.5), kHighsInf);
        status = highs->run();
    }

    if (ht <= -_instance->jobs) {
        std::cout << "Error in Pricing MIP: could not find good reduced cost" << std::endl;
        return tmp; // return standard pricing solution
	}

    solution.clear();

    if (highs->getModelStatus() == HighsModelStatus::kOptimal) {
        const auto& sol = highs->getSolution();

        for (int j = 0; j < _instance->jobs; ++j) {
            if (sol.col_value[j] > 0.5) {
                solution.push_back(j);
            }
        }
    }

    return tmp;
}

template <typename RmpSolver>
void TemplatePrice<RmpSolver>::init(tf::Executor* executor, RmpSolver* rmp, GapInstance* instance) {
    _rmp = rmp;
    _instance = instance;
	_executor = executor;
    _template.init(instance->machines, instance->jobs);
    _mip.reset(new PricingBlockVector<GapPricingMIP<RmpSolver>>(instance->machines));
    _mip->init(*instance);
}

template <typename RmpSolver>
double TemplatePrice<RmpSolver>::optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs) {
	tf::Taskflow taskflow;
    tf::IndexRange range(0, _instance->machines, 1);

    taskflow.for_each_by_index(range, [&](tf::IndexRange<int> subrange) {
        std::vector<double> obj(_instance->jobs);

        for (int m = subrange.begin(); m < subrange.end(); m += subrange.step_size()) {
            for (int j = 0; j < _instance->jobs; ++j) {
                obj[j] = duals[j] - _instance->costs[m][j];
            }

            reduced_costs[m] = _mip->_pricing[m].optimize_template(_template[m], obj, duals[_instance->jobs + m]);
            pricing[m].solution.swap(_mip->_pricing[m].solution);
            pricing[m].solution.push_back(_instance->jobs + m);
        }
    });

	_executor->run(std::move(taskflow)).wait();

    double optimal_pricing = 0.0;

    for (int m = 0; m < _instance->machines; ++m)
        optimal_pricing += reduced_costs[m] > 1e-6 ? reduced_costs[m] : 0.0;

    return optimal_pricing;
}

template class TemplatePrice<Highs>;

#ifdef SUPPORT_GUROBI

template class TemplatePrice<GurobiHighs>;

#endif