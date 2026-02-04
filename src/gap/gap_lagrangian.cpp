#include "gap_lagrangian.h"
#include "gap_instance.h"
#include "extern/scip/scip_knapsack.h"
#include "gap_pricing.h"
#include "utils.h"
#include "taskflow/taskflow.hpp"
#include "extern/dlib/optimization.h"

using Eigen::VectorXd;

class LagrangianProblem {
public:
    tf::Executor* _executor;
    const GapInstance& instance;
	PricingBlockVector<GapPricing> pricing;
    VectorXd gradient;
	double upper_bound;
    Timer cg_time;

    LagrangianProblem(const GapInstance& inst, tf::Executor& executor) : instance(inst), pricing(inst.machines), _executor(&executor) { 
		pricing.init(instance);
		upper_bound = std::numeric_limits<double>::infinity();
    }

    double operator()(const VectorXd& multipliers) {
		cg_time.start();
        tf::Taskflow taskflow;
        tf::IndexRange range(0, instance.machines, 1);

		VectorXd values = VectorXd::Zero(instance.machines);

        // solve knapsack subproblems for each machine
        taskflow.for_each_by_index(range, [&](tf::IndexRange<int> subrange) {
            std::vector<double> obj(instance.jobs);

            for (int m = subrange.begin(); m < subrange.end(); m += subrange.step_size()) {
                const std::vector<double>& costs = instance.costs[m];

                for (int j = 0; j < instance.jobs; ++j) {
                    obj[j] = multipliers[j] - costs[j];
                }

                values[m] = pricing[m].optimize(obj, 0);
            }
        });

        _executor->run(std::move(taskflow)).wait();

        // we negate because L-BFGS minimizes, but we want to maximize
        gradient = VectorXd::Constant(multipliers.size(), -1);

        for (int m = 0; m < instance.machines; ++m) {
            for (int j : pricing[m].solution) {
                ++gradient[j];
            }
        }

		// check for trivial upper bound (i.e., gradient is non-negative)
		if (gradient.minCoeff() > -1e-6) {
			// Note: there might be duplicates, so we need to take the minimum profit per job
            VectorXd upper = VectorXd::Constant(instance.jobs, std::numeric_limits<double>::infinity());

			for (int m = 0; m < instance.machines; ++m) {
				const auto& costs = instance.costs[m];

				for (int j : pricing[m].solution) {
                    if (upper[j] > costs[j]) {
                        upper[j] = costs[j];
					}
				}
			}

            upper_bound = std::min(upper_bound, upper.sum());
		}
		cg_time.pause();

        // update objective
        return values.sum() - multipliers.sum();
    }
};

void GapLagrangian::solve(quill::CsvWriter<CsvSchema, quill::FrontendOptions>& csv_writer) {
    Timer timer;
    ColumnAlignOutput tbl;

    tbl.add_column("#Its", 7);
    tbl.add_column("LB", 14, 2);
    tbl.add_column("UB", 14, 2);
    tbl.add_column("fDelta", 14, 5);
    tbl.add_column("Time", 10, 1);
    tbl.write_header();

    const int ITERATION_OUTPUT = 5;
    const double ITERATION_TIME = 1.0;
    double previous_logging_time = -1;
	bool should_stop = false;

    HandleCtrlC ctrl_c_handler([&]() {
        if (should_stop) exit(-1);  // force stop

        std::cout << std::format("{} Lagrangian Relaxation: Ctrl-C pressed, stopping...\n", instance.name);
        should_stop = true;
    });

    LagrangianProblem problem(instance, _executor);

    auto callback = [&](int num_iterations, double obj, double f_delta) {
        if (num_iterations % ITERATION_OUTPUT == 0 && timer.TotalSeconds() - previous_logging_time > ITERATION_TIME) {
            previous_logging_time = timer.TotalSeconds();
            tbl.output(num_iterations, -obj, problem.upper_bound, f_delta, previous_logging_time);
        }

        double lb = std::ceil(-obj - 1e-6);
        double gap = (problem.upper_bound - lb) / problem.upper_bound;

        csv_writer.append_row(instance.name, instance.name[0], instance.machines, instance.jobs, "lr", params.solver, "na", params.replication, num_iterations,
            -obj, problem.upper_bound, gap * 100, -obj, 0, 0, 0, timer.TotalSeconds() - problem.cg_time.TotalSeconds(), problem.cg_time.TotalSeconds(), timer.TotalSeconds(),
            "", "", "", 1, 0, "", 0);

        return (timer.TotalSeconds() > params.time_limit) || should_stop;
    };

    Eigen::VectorXd x = Eigen::VectorXd::Zero(instance.jobs);

    auto func = [&](const Eigen::VectorXd& x) { return problem(x); };
	auto grad = [&](const Eigen::VectorXd& x) { return problem.gradient; };

	auto search_strategy = dlib::lbfgs_search_strategy(256);
	auto stop_strategy = dlib::objective_delta_stop_strategy(&callback, 1e-6);

    _bound = -dlib::find_min(search_strategy, stop_strategy, func, grad, x);
    timer.pause();

    // store and output results
    _multipliers.resize(instance.jobs);

    for (int j = 0; j < instance.jobs; ++j) {
        _multipliers[j] = x[j];
    }

    tbl.output(stop_strategy._cur_iter, _bound, problem.upper_bound, "", timer.TotalSeconds());

    _bound = std::ceil(_bound - 1e-6);
    tbl.output(stop_strategy._cur_iter, _bound, problem.upper_bound, "", timer.TotalSeconds());

    // final entry
    double gap = (problem.upper_bound - _bound) / problem.upper_bound;

    csv_writer.append_row(instance.name, instance.name[0], instance.machines, instance.jobs, "lr", params.solver, "na", params.replication, stop_strategy._cur_iter,
        _bound, problem.upper_bound, gap * 100, _bound, 0, 0, 0, timer.TotalSeconds() - problem.cg_time.TotalSeconds(), problem.cg_time.TotalSeconds(), timer.TotalSeconds(),
        "", "", "", 1, 0, "", 1);

    std::cout << std::format("\n"
        "Inst: {}\n"
        "Time: {:.3f} s\n"
        "#Its: {}\n",
        instance.name, timer.TotalSeconds(), stop_strategy._cur_iter);
}
