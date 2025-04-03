#pragma once
#include <string>
#include "parameters.h"
#include "block/column_generation.h"
#include "gap_compact.h"
#include "quill/CsvWriter.h"
#include "quill/core/FrontendOptions.h"
#include "extern/scip/scip_knapsack.h"
#include "highs/parallel/HighsParallel.h"
#include "highs/Highs.h"

struct CsvSchema {
    static constexpr char const* header = "instance,algorithm,nodes,unexplored,iterations,lb,ub,gap,obj,redcost,basis,cols,rmptime,cgtime,time,lpiters,delta,last";
    static constexpr char const* format = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}";
};

struct GapPricing {
    size_t _machine = 0;
    const GapInstance* _instance = nullptr;
    std::vector<int> solution;
    std::vector<bool> lb_bound;

    void init(uint32_t index, const GapInstance& instance) {
		_instance = &instance;
		_machine = index;
    }

    double optimize(const std::vector<double>& obj, double offset) {
        double scip_opt;
        int solution_size = 0;
        solution.resize(_instance->jobs);
        SCIPsolveKnapsackExactly(_instance->jobs, _instance->demands[_machine].data(), obj.data(), _instance->capacity[_machine], solution.data(), &solution_size, &scip_opt);
        solution.resize(solution_size);
        
        return scip_opt + offset;
    }
};

struct GapPricingMIP : public GapPricing {
    std::unique_ptr<Highs> highs;

    void init(uint32_t index, const GapInstance& instance) {
		GapPricing::init(index, instance);

        // max tx
        //   0 <= wx <= C
        // eps <= dx <= inf
        //  lb <= tx <= ub
		std::vector<std::vector<double>> constraints(3, std::vector<double>());
		std::copy(instance.demands[index].begin(), instance.demands[index].end(), std::back_inserter(constraints[0]));

        HighsModel model = construct_system(constraints, {
            { 0, _instance->capacity[_machine] }, 
            { 0, kHighsInf }, 
            { 0, kHighsInf } 
        });

        highs.reset(new Highs);
        highs->setOptionValue("output_flag", false);
        //highs->setOptionValue("threads", 1);
        highs->passModel(model);
    }

    double optimize_template(const std::vector<double>& template_obj, const std::vector<double>& duals, double offset) {
        // heuristic dual & template
        double tmp = optimize(duals, offset);
        if (tmp <= 1e-3) {
            return tmp;
        }

        highs->changeColsCost(0, _instance->jobs - 1, template_obj.data());
        highs->changeRowBounds(1, -offset + 1e-3, kHighsInf);  // 1e-3 tol to avoid numerical issues
        highs->changeRowBounds(2, -_instance->jobs, kHighsInf);

        for (int j = 0; j < _instance->jobs; ++j)
            highs->changeCoeff(1, j, duals[j]);

        for (int j = 0; j < _instance->jobs; ++j)
            highs->changeCoeff(2, j, template_obj[j]);

        highs->run();
        double ht = highs->getObjectiveValue(), hd = offset;

        // switch objective, add new constraint
        HighsStatus status = highs->changeColsCost(0, _instance->jobs - 1, duals.data());
        highs->changeRowBounds(2, std::ceil(ht - 0.5), kHighsInf);
        status = highs->run();

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
};



struct DantzigPrice {
    Highs* _rmp = nullptr;
    GapInstance* _instance = nullptr;
    static constexpr const char* name = "Dantzig";

    void init(Highs* rmp, GapInstance* instance) {
        _rmp = rmp;
        _instance = instance;
    }

    void update() { }
    void init_feasible() { }
    void debug(std::string algorithm) {  }
    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs);
};


struct WentgesTemplatePrice {
    Highs* _rmp = nullptr;
    GapInstance* _instance = nullptr;
    TemplatePricing _template;

    static constexpr const char* name = "WentgesTemplate";
    std::vector<double> alpha;

    void init(Highs* rmp, GapInstance* instance) {
        _rmp = rmp;
        _instance = instance;
        _template.init(instance->machines, instance->jobs);
        alpha.resize(instance->machines, 1);
    }

    void update() {
        _template.update(_rmp->getSolution(), *_rmp);
    }

    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs, bool update_duals = true);

    void debug(std::string algorithm) { }
    void init_feasible() { }
};

struct WentgesPrice {
    Highs* _rmp = nullptr;
    GapInstance* _instance = nullptr;
    static constexpr const char* name = "Wentges";

    DantzigPrice dantzig;

    double best_reduced_cost = kHighsInf;
    double alpha = 0.0;
    size_t mis_price = 0;

    std::vector<double> stabilized_duals_for_pricing_old_columns;
    std::vector<double> stabilized_duals_for_pricing;

    std::vector<double> lagrangian_constraint_values;
    std::vector<double> subgradient;

    void init(Highs* rmp, GapInstance* instance) {
        dantzig.init(rmp, instance);
        _rmp = rmp;
        _instance = instance;

        stabilized_duals_for_pricing_old_columns.resize(rmp->getNumRow(), 0);
        stabilized_duals_for_pricing.resize(rmp->getNumRow(), 0);
        lagrangian_constraint_values.resize(rmp->getNumRow(), 0);
        subgradient.resize(rmp->getNumRow(), 0);
    }

    void init_feasible() {
        stabilized_duals_for_pricing = _rmp->getSolution().row_dual;
    }

    void update() {
        if (_rmp->getObjectiveValue() < best_reduced_cost) {
            best_reduced_cost = _rmp->getObjectiveValue();
            stabilized_duals_for_pricing_old_columns = stabilized_duals_for_pricing;
            //alpha = 0.9;
        }
    }

    void debug(std::string algorithm) {   }
    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs);
};


struct TemplatePrice {
    Highs* _rmp = nullptr;
    GapInstance* _instance = nullptr;
    TemplatePricing _template;
    static constexpr const char* name = "Template";
    std::unique_ptr<PricingBlockVector<GapPricingMIP>> _mip;

	TemplatePrice() {}

    TemplatePrice(const TemplatePrice& copy) {
		_rmp = copy._rmp;
		_instance = copy._instance;
		_template = copy._template;
		_mip = std::make_unique<PricingBlockVector<GapPricingMIP>>(copy._instance->machines);
		_mip->init(*_instance);
    }

    void init(Highs* rmp, GapInstance* instance) {
        _rmp = rmp;
        _instance = instance;
        _template.init(instance->machines, instance->jobs);
		_mip.reset(new PricingBlockVector<GapPricingMIP>(instance->machines));
        _mip->init(*instance);
    }

    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs) {
        highs::parallel::for_each(0, _instance->machines, [&](HighsInt start, HighsInt end) {
            for (int m = start; m < end; ++m) {
                std::vector<double> obj(_instance->jobs);

                for (int j = 0; j < _instance->jobs; ++j) {
                    obj[j] = duals[j] - _instance->profit[m][j];
                }

                reduced_costs[m] = _mip->_pricing[m].optimize_template(_template[m], obj, duals[_instance->jobs + m]);
				pricing[m].solution.swap(_mip->_pricing[m].solution);
                pricing[m].solution.push_back(_instance->jobs + m);
            }
        });

        double optimal_pricing = 0.0;

        for (int m = 0; m < _instance->machines; ++m)
            optimal_pricing += reduced_costs[m] > 1e-6 ? reduced_costs[m] : 0.0;

        return optimal_pricing;
    }

    void update() {
        _template.update(_rmp->getSolution(), *_rmp);
    }

    void init_feasible() {}
    void debug(std::string algorithm) {}
};



struct FixedTemplatePrice {
    Highs* _rmp = nullptr;
    GapInstance* _instance = nullptr;
    TemplatePricing _template;
    static constexpr const char* name = "FixedTemplate";
    std::unique_ptr<PricingBlockVector<GapPricingMIP>> _mip;

	FixedTemplatePrice() {}

    FixedTemplatePrice(const FixedTemplatePrice& copy) {
        _rmp = copy._rmp;
        _instance = copy._instance;
        _template = copy._template;
        _mip = std::make_unique<PricingBlockVector<GapPricingMIP>>(copy._instance->machines);
        _mip->init(*_instance);
    }

    void init(Highs* rmp, GapInstance* instance) {
        _rmp = rmp;
        _instance = instance;
        _template.init(instance->machines, instance->jobs);
        _mip.reset(new PricingBlockVector<GapPricingMIP>(instance->machines));
        _mip->init(*instance);
    }

    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs) {
        highs::parallel::for_each(0, _instance->machines, [&](HighsInt start, HighsInt end) {
            std::vector<double> obj(_instance->jobs);

            for (int m = start; m < end; ++m) {
                for (int j = 0; j < _instance->jobs; ++j) {
                    obj[j] = duals[j] - _instance->profit[m][j];
                }

                reduced_costs[m] = _mip->_pricing[m].optimize_template(_template[m], obj, duals[_instance->jobs + m]);
                pricing[m].solution.swap(_mip->_pricing[m].solution);
                pricing[m].solution.push_back(_instance->jobs + m);
            }
        });

        double optimal_pricing = 0.0;

        for (int m = 0; m < _instance->machines; ++m)
            optimal_pricing += reduced_costs[m] > 1e-6 ? reduced_costs[m] : 0.0;

        return optimal_pricing;
    }

    void debug(std::string algorithm) {   }
    void init_feasible() { }
    void update() { }
};


struct DualColumnManagement {
    Highs* _rmp = nullptr;
    GapInstance* _instance = nullptr;

    void init(Highs* rmp, GapInstance* instance) {
        _rmp = rmp;
        _instance = instance;
    }

    void calculate_reduced_costs(Highs& h, const std::vector<double>& duals, std::vector<double>& reduced_costs) {
        auto& lp = h.getLp();
        assert(lp.a_matrix_.isColwise() == true);

        for (int i = 0; i < lp.num_col_; ++i) {
            for (int el = lp.a_matrix_.start_[i]; el < lp.a_matrix_.start_[i + 1]; ++el) {
                reduced_costs[i] += lp.a_matrix_.value_[el] * duals[lp.a_matrix_.index_[el]];
            }

            reduced_costs[i] -= lp.col_cost_[i];
        }
    }

    void reduce(size_t iteration_count);
};


struct TemplateFarkas {
    GapInstance* _instance;
    TemplatePricing _template;
    std::unique_ptr<PricingBlockVector<GapPricingMIP>> _mip;

    template <typename Pricer>
    void init(Pricer& pricer, gap_compact& lp) {
        _instance = pricer._instance;
        _template.init(_instance->machines, _instance->jobs);
        _mip.reset(new PricingBlockVector<GapPricingMIP>(instance->machines));
        _mip->init(*instance);

        const auto& solution = lp._solution;

        for (int m = 0; m < _instance->machines; ++m) {
            for (int j = 0; j < _instance->jobs; ++j) {
                double value = solution.col_value[m * _instance->jobs + j];
                _template._template_columns[m][j] = (value > 1 - 1e-6) - (value < 1e-6);
            }
        }
    }

    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs) {
        highs::parallel::for_each(0, _instance->machines, [&](HighsInt start, HighsInt end) {
            for (int m = start; m < end; ++m) {
                reduced_costs[m] = _mip->_pricing[m].optimize_template(_template[m], duals, duals[_instance->jobs + m]);
                pricing[m].solution.swap(_mip->_pricing[m].solution);
                pricing[m].solution.push_back(_instance->jobs + m);
            }
        });

        return 0;
    }
};

struct HeuristicTemplateFarkas {
    GapInstance* _instance = nullptr;
    TemplatePricing _template;
    std::vector<double> dual_values;
    std::unique_ptr<PricingBlockVector<GapPricingMIP>> _mip;

    template <typename Pricer>
    void init(Pricer& pricer, gap_compact& lp) {
        _instance = pricer._instance;
        _template.init(_instance->machines, _instance->jobs);
        _mip.reset(new PricingBlockVector<GapPricingMIP>(_instance->machines));
        _mip->init(*_instance);

        for (int m = 0; m < _instance->machines; ++m) {
            for (int j = 0; j < _instance->jobs; ++j) {
                double value = lp._solution.col_value[m * _instance->jobs + j];
                _template._template_columns[m][j] = (value > 1 - 1e-6) - (value < 1e-6);
            }
        }
    }

    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs) {
        highs::parallel::for_each(0, _instance->machines, [&](HighsInt start, HighsInt end) {
            for (int m = start; m < end; ++m) {
                double obj = _mip->_pricing[m].optimize_template(_template[m], duals, duals[_instance->jobs + m]);
                pricing[m].solution.swap(_mip->_pricing[m].solution);
                pricing[m].solution.push_back(_instance->jobs + m);
                reduced_costs[m] = obj != -kHighsInf ? 1 : -kHighsInf;
            }
        });

        double feasible = 0;
		for (int m = 0; m < _instance->machines; ++m) {
            feasible = std::min(feasible, reduced_costs[m]);
		}

        return feasible;
    }
};

struct FixedTemplateFarkas {
    GapInstance* _instance = nullptr;
    TemplatePricing _template;
    std::unique_ptr<PricingBlockVector<GapPricingMIP>> _mip;

    template <typename Pricer>
    void init(Pricer& pricer, gap_compact& lp) {
        _instance = pricer._instance;
        _mip.reset(new PricingBlockVector<GapPricingMIP>(_instance->machines));
        _mip->init(*_instance);
    }

    template <typename Pricer>
    void init_fractional(Pricer& pricer, gap_compact& lp) {
        _instance = pricer._instance;
        _template.init(_instance->machines, _instance->jobs);
    }

    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs) {
        highs::parallel::for_each(0, _instance->machines, [&](HighsInt start, HighsInt end) {
            for (int m = start; m < end; ++m) {
                double obj = _mip->_pricing[m].optimize_template(_template[m], duals, duals[_instance->jobs + m]);
                pricing[m].solution.swap(_mip->_pricing[m].solution);
                pricing[m].solution.push_back(_instance->jobs + m);
                reduced_costs[m] = obj != -kHighsInf ? 1 : -kHighsInf;
            }
        });

        double feasible = 0;
        for (int m = 0; m < _instance->machines; ++m) {
            feasible = std::min(feasible, reduced_costs[m]);
        }

        return feasible;
    }
};



struct DantzigFarkas {
    GapInstance* _instance = nullptr;
    Highs* _rmp = nullptr;
    TemplatePricing _template; // not used, but compile hack for now

    template <typename Pricer>
    void init(Pricer& pricer, gap_compact& lp) {
        _instance = pricer._instance;
        _template.init(_instance->machines, _instance->jobs);
    }

    template <typename Pricer>
    void init_fractional(Pricer& pricer, gap_compact& lp) {
        init(pricer, lp);
    }

    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs) {
        highs::parallel::for_each(0, _instance->machines, [&](HighsInt start, HighsInt end) {
            for (int m = start; m < end; ++m) {
                reduced_costs[m] = pricing[m].optimize(duals, duals[_instance->jobs + m]);
                pricing[m].solution.push_back(_instance->jobs + m);
            }
        });

        return 0;
    }
};

struct WentgesTemplateFarkas {
    GapInstance* _instance = nullptr;
    Highs* _rmp = nullptr;
	WentgesTemplatePrice _template; // not used, but compile hack for now

    template <typename Pricer>
    void init(Pricer& pricer, gap_compact& lp) {
        _instance = pricer._instance;
        _template.init(pricer._rmp, _instance);

        for (int m = 0; m < _instance->machines; ++m) {
            for (int j = 0; j < _instance->jobs; ++j) {
                double value = lp._solution.col_value[m * _instance->jobs + j];
                _template._template._template_columns[m][j] = (value > 1 - 1e-6) - (value < 1e-6);
            }
        }
    }

    template <typename Pricer>
    void init_fractional(Pricer& pricer, gap_compact& lp) {
        init(pricer, lp);
    }

    double optimize(const std::vector<double>& duals, PricingBlockVector<GapPricing>& pricing, std::vector<double>& reduced_costs) {
		_template.optimize(duals, pricing, reduced_costs, false);
        return 0;
    }
};

struct OpenNode {
    double lower_bound = -kHighsInf;
    bool lpOptimal = false;
    std::vector<HighsInt> fixed_lb;
};

struct GapSolver {
    static constexpr int ITERATION_OUTPUT = 5;
    static constexpr double ITERATION_TIME = 1.0;

    int basis_size = 0;
	bool isIntegral = false;
    size_t iteration_count = 0;
    size_t lp_iteration_count = 0;
	size_t fractional_count = 0;
	size_t prune_count = 0;
	size_t leaf_count = 0;
    double _LB = -kHighsInf;
    double _UB = kHighsInf;
    double previous_logging_time = -1;
    Timer total_time, rmp_time, cg_time;

    std::unique_ptr<Highs> rmp;

    Parameters& params;
    GapInstance instance;

    quill::CsvWriter<CsvSchema, quill::FrontendOptions>& csv_writer;
    std::vector<double> _compact_solution;
    std::vector<double> _compact_solution_best;
    std::vector<double> _ones;
    std::vector<double> _reduced_costs;

    ColumnAlignOutput tbl;
    DualColumnManagement column_management;
    PricingBlockVector<GapPricing> pricing;
    gap_compact lp;

    GapSolver(std::string filename, Parameters& params, quill::CsvWriter<CsvSchema, quill::FrontendOptions>& csv_writer)
        : instance(filename), params(params), csv_writer(csv_writer), pricing(instance.machines), lp(instance) {

        _ones.assign(instance.jobs + 1, 1);
        _reduced_costs.assign(instance.machines, 0);
		_compact_solution.assign(instance.jobs * instance.machines, 0);
		_compact_solution_best.assign(instance.jobs * instance.machines, 0);

		tbl.show_output = params.show_output;
        tbl.add_column("#Nodes", 7);
        tbl.add_column("#UnExp", 7);
        tbl.add_column("#Its", 7);
        tbl.add_column("LB", 14, 2);
        tbl.add_column("UB", 14, 2);
        tbl.add_column("Gap%", 9, 2);
        tbl.add_column("OBJ", 14, 2);
        tbl.add_column("Dfeas", 14, 2);
        tbl.add_column("#Basis", 10, 0);
        tbl.add_column("#Cols", 10, 0);
        tbl.add_column("Time", 10, 1);
        tbl.add_column("Lp #Its", 11);
        tbl.add_column("#Frac", 7);
        tbl.add_column("#Prune", 7);
        tbl.add_column("#Leaf", 7);
    }

    // used for branching
    std::vector<OpenNode> openNodes;

    void presolve();

    template <typename PricerType, typename FarkasPricerType> void solve(PricerType& pricer, FarkasPricerType& pricer_farkas);

    template <typename PricerType, typename FarkasPricerType>
    void solve() {
        PricerType pricer;
        FarkasPricerType pricer_farkas;
		solve(pricer, pricer_farkas);
    }

    template <typename FarkasPricerType>
    bool restoreFeasibility(FarkasPricerType &pricer_farkas);
	void updateCompactSolution();
};