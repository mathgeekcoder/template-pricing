// Copyright (C) 2008  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
//
// Modified by Luke Marshall to use eigen and extract relevant parts only
#pragma once

#include <cmath>
#include <limits>
#include "optimization_search_strategies.h"
#include "optimization_stop_strategies.h"
#include "optimization_line_search.h"

namespace dlib
{
	static inline auto clamp(const Eigen::VectorXd& v, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub) {
		return v.cwiseMin(ub).cwiseMax(lb);
	}


	template <typename funct>
	struct clamped_function_object
	{
		clamped_function_object(
			const funct& f_,
			const Eigen::VectorXd& x_lower_,
			const Eigen::VectorXd& x_upper_
		) : f(f_), x_lower(x_lower_), x_upper(x_upper_)
		{
		}

		template <typename T>
		double operator() (
			const T& x
			) const
		{
			return f(clamp(x, x_lower, x_upper));
		}

		const funct& f;
		const Eigen::VectorXd& x_lower;
		const Eigen::VectorXd& x_upper;
	};

	template <typename funct>
	clamped_function_object<funct> clamp_function(
		const funct& f,
		const Eigen::VectorXd& x_lower,
		const Eigen::VectorXd& x_upper
	) {
		return clamped_function_object<funct>(f, x_lower, x_upper);
	}


	template <typename T, typename U, typename V>
	T zero_bounded_variables(const double eps, T vect, const T& x, const T& gradient, const U& x_lower, const V& x_upper) {

		for (long i = 0; i < gradient.size(); ++i) {
			const double tol = eps * std::abs(x(i));
			// if x(i) is an active bound constraint
			if (x_lower(i) + tol >= x(i) && gradient(i) > 0)
				vect(i) = 0;
			else if (x_upper(i) - tol <= x(i) && gradient(i) < 0)
				vect(i) = 0;
		}
		return vect;
	}

	// ----------------------------------------------------------------------------------------

	template <typename T, typename U, typename V>
	T gap_step_assign_bounded_variables(
		const double eps,
		T vect,
		const T& x,
		const T& gradient,
		const U& x_lower,
		const V& x_upper
	)
	{
		for (long i = 0; i < gradient.size(); ++i)
		{
			const double tol = eps * std::abs(x(i));
			// If x(i) is an active bound constraint then we should set its search
			// direction such that a single step along the direction either does nothing or
			// closes the gap of size tol before hitting the bound exactly.
			if (x_lower(i) + tol >= x(i) && gradient(i) > 0)
				vect(i) = x_lower(i) - x(i);
			else if (x_upper(i) - tol <= x(i) && gradient(i) < 0)
				vect(i) = x_upper(i) - x(i);
		}
		return vect;
	}

	template <
		typename search_strategy_type,
		typename stop_strategy_type,
		typename funct,
		typename funct_der>
	double find_min_box_constrained(
		search_strategy_type search_strategy,
		stop_strategy_type& stop_strategy,
		const funct& f,
		const funct_der& der,
		Eigen::VectorXd& x,
		const Eigen::VectorXd& x_lower,
		const Eigen::VectorXd& x_upper
	) {
		Eigen::VectorXd g, s;
		double f_value = f(x);
		g = der(x);

		if (!std::isfinite(f_value))
			throw std::runtime_error("The objective function generated non-finite outputs");

		if (!std::isfinite(g.cwiseAbs().maxCoeff()))
			throw std::runtime_error("The objective function generated non-finite outputs");

		// gap_eps determines how close we have to get to a bound constraint before we
		// start basically dropping it from the optimization and consider it to be an
		// active constraint.
		const double gap_eps = 1e-8;

		double last_alpha = 1;
		while (stop_strategy.should_continue_search(x, f_value, g))
		{
			s = search_strategy.get_next_direction(x, f_value, zero_bounded_variables(gap_eps, g, x, g, x_lower, x_upper));
			s = gap_step_assign_bounded_variables(gap_eps, s, x, g, x_lower, x_upper);

			double alpha = backtracking_line_search(
				make_line_search_function(clamp_function(f, x_lower, x_upper), x, s, f_value),
				f_value,
				g.dot(s), // compute gradient for the line search
				last_alpha,
				search_strategy.get_wolfe_rho(),
				search_strategy.get_max_line_search_iterations());

			// Do a trust region style thing for alpha.  The idea is that if we take a
			// small step then we are likely to take another small step.  So we reuse the
			// alpha from the last iteration unless the line search didn't shrink alpha at
			// all, in that case, we start with a bigger alpha next time.
			if (alpha == last_alpha)
				last_alpha = std::min(last_alpha * 10, 1.0);
			else
				last_alpha = alpha;

			// Take the search step indicated by the above line search
			x = clamp(x + alpha * s, x_lower, x_upper);
			g = der(x);

			if (!std::isfinite(f_value))
				throw std::runtime_error("The objective function generated non-finite outputs");

			if (!std::isfinite(g.cwiseAbs().maxCoeff()))
				throw std::runtime_error("The objective function generated non-finite outputs");
		}

		return f_value;
	}

	// ----------------------------------------------------------------------------------------

	template <
		typename search_strategy_type,
		typename stop_strategy_type,
		typename funct,
		typename funct_der
	>
	double find_min_box_constrained(
		search_strategy_type search_strategy,
		stop_strategy_type& stop_strategy,
		const funct& f,
		const funct_der& der,
		Eigen::VectorXd& x,
		double x_lower = -std::numeric_limits<double>::infinity(),
		double x_upper = std::numeric_limits<double>::infinity()
	) {
		Eigen::VectorXd lb = Eigen::VectorXd::Constant(x.size(), x_lower);
		Eigen::VectorXd ub = Eigen::VectorXd::Constant(x.size(), x_upper);

		return find_min_box_constrained(search_strategy, stop_strategy, f, der, x, lb, ub);
	}


	template <
		typename search_strategy_type,
		typename stop_strategy_type,
		typename funct,
		typename funct_der
	>
	double find_min(
		search_strategy_type search_strategy,
		stop_strategy_type& stop_strategy,
		const funct& f,
		const funct_der& der,
		Eigen::VectorXd& x,
		double min_f = -std::numeric_limits<double>::infinity()
	) {
		Eigen::VectorXd g, s;

		double f_value = f(x);
		g = der(x);

		if (!std::isfinite(f_value))
			throw std::runtime_error("The objective function generated non-finite outputs");
		if (!std::isfinite(g.cwiseAbs().maxCoeff()))
			throw std::runtime_error("The objective function generated non-finite outputs");

		while (stop_strategy.should_continue_search(x, f_value, g) && f_value > min_f)
		{
			s = search_strategy.get_next_direction(x, f_value, g);

			double alpha = line_search(
				make_line_search_function(f, x, s, f_value),
				f_value,
				make_line_search_function(der, x, s, g),
				g.dot(s), // compute initial gradient for the line search
				search_strategy.get_wolfe_rho(), search_strategy.get_wolfe_sigma(), min_f,
				search_strategy.get_max_line_search_iterations());

			// Take the search step indicated by the above line search
			x += alpha * s;

			if (!std::isfinite(f_value))
				throw std::runtime_error("The objective function generated non-finite outputs");
			if (!std::isfinite(g.cwiseAbs().maxCoeff()))
				throw std::runtime_error("The objective function generated non-finite outputs");
		}

		return f_value;
	}
}

