// Copyright (C) 2008  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
//
// Modified by Luke Marshall to use eigen and extract relevant parts only
#pragma once

#include <cmath>
#include <limits>
#include <utility>
#include <cassert>
#include <Eigen/Core>

namespace dlib
{
	template <typename T>
	T put_in_range(const T a, const T b, const T val)
		/*!
			requires
				- T is a type that looks like double, float, int, or so forth
			ensures
				- if (val is within the range [a,b]) then
					- returns val
				- else
					- returns the end of the range [a,b] that is closest to val
		!*/
	{
		if (a < b) {
			if (val < a)
				return a;
			else if (val > b)
				return b;
		}
		else {
			if (val < b)
				return b;
			else if (val > a)
				return a;
		}

		return val;
	}

	// overload for double 
	inline double put_in_range(const double a, const double b, const double val) {
		return put_in_range<double>(a, b, val);
	}

	template <typename funct, typename T>
	class line_search_funct
	{
	public:
		line_search_funct(const funct& f_, const T& start_, const T& direction_)
			: f(f_), start(start_), direction(direction_), matrix_r(0), scalar_r(0)
		{
		}

		line_search_funct(const funct& f_, const T& start_, const T& direction_, T& r)
			: f(f_), start(start_), direction(direction_), matrix_r(&r), scalar_r(0)
		{
		}

		line_search_funct(const funct& f_, const T& start_, const T& direction_, double& r)
			: f(f_), start(start_), direction(direction_), matrix_r(0), scalar_r(&r)
		{
		}

		double operator()(const double& x) const {
			return get_value(f(start + x * direction));
		}

	private:
		double get_value(const double& r) const
		{
			// save a copy of this value for later
			if (scalar_r)
				*scalar_r = r;

			return r;
		}

		template <typename U>
		double get_value(const U& r) const
		{
			// save a copy of this value for later
			if (matrix_r)
				*matrix_r = r;

			return r.dot(direction);
		}

		const funct& f;
		const T& start;
		const T& direction;
		T* matrix_r;
		double* scalar_r;
	};

	template <typename funct, typename T>
	const line_search_funct<funct, T> make_line_search_function(const funct& f, const T& start, const T& direction) {
		return line_search_funct<funct, T>(f, start, direction);
	}

	// ----------------------------------------------------------------------------------------

	template <typename funct, typename T>
	const line_search_funct<funct, T> make_line_search_function(const funct& f, const T& start, const T& direction, double& f_out) {
		return line_search_funct<funct, T>(f, start, direction, f_out);
	}

	// ----------------------------------------------------------------------------------------

	template <typename funct, typename T>
	const line_search_funct<funct, T> make_line_search_function(const funct& f, const T& start, const T& direction, T& grad_out) {
		return line_search_funct<funct, T>(f, start, direction, grad_out);
	}

	inline double poly_min_extrap(
		double f0,
		double d0,
		double f1,
		double d1,
		double limit = 1
	)
	{
		const double n = 3 * (f1 - f0) - 2 * d0 - d1;
		const double e = d0 + d1 - 2 * (f1 - f0);

		// find the minimum of the derivative of the polynomial
		double temp = std::max(n * n - 3 * e * d0, 0.0);

		if (temp < 0)
			return 0.5;

		temp = std::sqrt(temp);

		if (std::abs(e) <= std::numeric_limits<double>::epsilon())
			return 0.5;

		// figure out the two possible min values
		double x1 = (temp - n) / (3 * e);
		double x2 = -(temp + n) / (3 * e);

		// compute the value of the interpolating polynomial at these two points
		double y1 = f0 + d0 * x1 + n * x1 * x1 + e * x1 * x1 * x1;
		double y2 = f0 + d0 * x2 + n * x2 * x2 + e * x2 * x2 * x2;

		// pick the best point
		double x;
		if (y1 < y2)
			x = x1;
		else
			x = x2;

		// now make sure the minimum is within the allowed range of [0,limit] 
		return put_in_range(0, limit, x);
	}

	// ----------------------------------------------------------------------------------------

	inline double poly_min_extrap(
		double f0,
		double d0,
		double f1
	)
	{
		const double temp = 2 * (f1 - f0 - d0);
		if (std::abs(temp) <= d0 * std::numeric_limits<double>::epsilon())
			return 0.5;

		const double alpha = -d0 / temp;

		// now make sure the minimum is within the allowed range of (0,1) 
		return put_in_range(0, 1, alpha);
	}

	// ----------------------------------------------------------------------------------------

	inline double poly_min_extrap(
		double f0,
		double d0,
		double x1,
		double f_x1,
		double x2,
		double f_x2
	)
	{
		assert(0 < x1 && x1 < x2); // "Invalid inputs were given to this function.

		// The contents of this function follow the equations described on page 58 of the
		// book Numerical Optimization by Nocedal and Wright, second edition.
		Eigen::Matrix<double, 2, 2> m;
		Eigen::Matrix<double, 2, 1> v;

		const double aa2 = x2 * x2;
		const double aa1 = x1 * x1;
		m << aa2, -aa1, -aa2 * x2, aa1* x1;
		v << f_x1 - f0 - d0 * x1, f_x2 - f0 - d0 * x2;


		double temp = aa2 * aa1 * (x1 - x2);

		// just take a guess if this happens
		if (temp == 0 || std::fpclassify(temp) == FP_SUBNORMAL)
		{
			return x1 / 2.0;
		}

		Eigen::Matrix<double, 2, 1> temp2;
		temp2 = m * v / temp;
		const double a = temp2(0);
		const double b = temp2(1);

		temp = b * b - 3 * a * d0;
		if (temp < 0 || a == 0)
		{
			// This is probably a line so just pick the lowest point
			if (f0 < f_x2)
				return 0;
			else
				return x2;
		}
		temp = (-b + std::sqrt(temp)) / (3 * a);
		return put_in_range(0, x2, temp);
	}

	template <
		typename funct,
		typename funct_der
	>
	double line_search(
		const funct& f,
		const double f0,
		const funct_der& der,
		const double d0,
		double rho,
		double sigma,
		double min_f,
		unsigned long max_iter
	) {
		assert(0 < rho && rho < sigma && sigma < 1 && max_iter > 0);

		// The bracketing phase of this function is implemented according to block 2.6.2 from
		// the book Practical Methods of Optimization by R. Fletcher.   The sectioning 
		// phase is an implementation of 2.6.4 from the same book.

		// 1 <= tau1a < tau1b. Controls the alpha jump size during the bracketing phase of
		// the search.
		const double tau1a = 1.4;
		const double tau1b = 9;

		// it must be the case that 0 < tau2 < tau3 <= 1/2 for the algorithm to function
		// correctly but the specific values of tau2 and tau3 aren't super important.
		const double tau2 = 1.0 / 10.0;
		const double tau3 = 1.0 / 2.0;


		// Stop right away and return a step size of 0 if the gradient is 0 at the starting point
		if (std::abs(d0) <= std::abs(f0) * std::numeric_limits<double>::epsilon())
			return 0;

		// Stop right away if the current value is good enough according to min_f
		if (f0 <= min_f)
			return 0;

		// Figure out a reasonable upper bound on how large alpha can get.
		const double mu = (min_f - f0) / (rho * d0);


		double alpha = 1;
		if (mu < 0)
			alpha = -alpha;
		alpha = put_in_range(0, 0.65 * mu, alpha);


		double last_alpha = 0;
		double last_val = f0;
		double last_val_der = d0;

		// The bracketing stage will find a range of points [a,b]
		// that contains a reasonable solution to the line search
		double a, b;

		// These variables will hold the values and derivatives of f(a) and f(b)
		double a_val, b_val, a_val_der, b_val_der;

		// This thresh value represents the Wolfe curvature condition
		const double thresh = std::abs(sigma * d0);

		unsigned long itr = 0;

		// do the bracketing stage to find the bracket range [a,b]
		while (true) {
			++itr;
			const double val = f(alpha);
			const double val_der = der(alpha);

			// we are done with the line search since we found a value smaller
			// than the minimum f value
			if (val <= min_f)
				return alpha;

			if (val > f0 + rho * alpha * d0 || val >= last_val)
			{
				a_val = last_val;
				a_val_der = last_val_der;
				b_val = val;
				b_val_der = val_der;

				a = last_alpha;
				b = alpha;
				break;
			}

			if (std::abs(val_der) <= thresh)
				return alpha;

			// if we are stuck not making progress then quit with the current alpha
			if (last_alpha == alpha || itr >= max_iter)
				return alpha;

			if (val_der >= 0)
			{
				a_val = val;
				a_val_der = val_der;
				b_val = last_val;
				b_val_der = last_val_der;

				a = alpha;
				b = last_alpha;
				break;
			}

			const double temp = alpha;
			// Pick a larger range [first, last].  We will pick the next alpha in that
			// range.
			double first, last;
			if (mu > 0)
			{
				first = std::min(mu, alpha + tau1a * (alpha - last_alpha));
				last = std::min(mu, alpha + tau1b * (alpha - last_alpha));
			}
			else
			{
				first = std::max(mu, alpha + tau1a * (alpha - last_alpha));
				last = std::max(mu, alpha + tau1b * (alpha - last_alpha));
			}

			// pick a point between first and last by doing some kind of interpolation
			if (last_alpha < alpha)
				alpha = last_alpha + (alpha - last_alpha) * poly_min_extrap(last_val, last_val_der, val, val_der, 1e10);
			else
				alpha = alpha + (last_alpha - alpha) * poly_min_extrap(val, val_der, last_val, last_val_der, 1e10);

			alpha = put_in_range(first, last, alpha);

			last_alpha = temp;

			last_val = val;
			last_val_der = val_der;
		}

		// Now do the sectioning phase from 2.6.4
		while (true)
		{
			++itr;
			double first = a + tau2 * (b - a);
			double last = b - tau3 * (b - a);

			// use interpolation to pick alpha between first and last
			alpha = a + (b - a) * poly_min_extrap(a_val, a_val_der, b_val, b_val_der);
			alpha = put_in_range(first, last, alpha);

			const double val = f(alpha);
			const double val_der = der(alpha);

			// we are done with the line search since we found a value smaller
			// than the minimum f value or we ran out of iterations.
			if (val <= min_f || itr >= max_iter)
				return alpha;

			// stop if the interval gets so small that it isn't shrinking any more due to rounding error 
			if (a == first || b == last)
			{
				return b;
			}

			// If alpha has basically become zero then just stop.  Think of it like this,
			// if we take the largest possible alpha step will the objective function
			// change at all?  If not then there isn't any point looking for a better
			// alpha.
			const double max_possible_alpha = std::max(std::abs(a), std::abs(b));
			if (std::abs(max_possible_alpha * d0) <= std::abs(f0) * std::numeric_limits<double>::epsilon())
				return alpha;


			if (val > f0 + rho * alpha * d0 || val >= a_val)
			{
				b = alpha;
				b_val = val;
				b_val_der = val_der;
			}
			else
			{
				if (std::abs(val_der) <= thresh)
					return alpha;

				if ((b - a) * val_der >= 0)
				{
					b = a;
					b_val = a_val;
					b_val_der = a_val_der;
				}

				a = alpha;
				a_val = val;
				a_val_der = val_der;
			}
		}
	}


	template <typename funct>
	double backtracking_line_search(
		const funct& f,
		double f0,
		double d0,
		double alpha,
		double rho,
		unsigned long max_iter
	) {
		assert(0 < rho && rho < 1 && max_iter > 0);

		// make sure alpha is going in the right direction.  That is, it should be opposite
		// the direction of the gradient.
		if ((d0 > 0 && alpha > 0) ||
			(d0 < 0 && alpha < 0))
		{
			alpha *= -1;
		}

		bool have_prev_alpha = false;
		double prev_alpha = 0;
		double prev_val = 0;
		unsigned long iter = 0;
		while (true)
		{
			++iter;
			const double val = f(alpha);
			if (val <= f0 + alpha * rho * d0 || iter >= max_iter)
			{
				return alpha;
			}
			else
			{
				// Interpolate a new alpha.  We also make sure the step by which we
				// reduce alpha is not super small.
				double step;
				if (!have_prev_alpha)
				{
					if (d0 < 0)
						step = alpha * put_in_range(0.1, 0.9, poly_min_extrap(f0, d0, val));
					else
						step = alpha * put_in_range(0.1, 0.9, poly_min_extrap(f0, -d0, val));
					have_prev_alpha = true;
				}
				else
				{
					if (d0 < 0)
						step = put_in_range(0.1 * alpha, 0.9 * alpha, poly_min_extrap(f0, d0, alpha, val, prev_alpha, prev_val));
					else
						step = put_in_range(0.1 * alpha, 0.9 * alpha, -poly_min_extrap(f0, -d0, -alpha, val, -prev_alpha, prev_val));
				}

				prev_alpha = alpha;
				prev_val = val;

				alpha = step;
			}
		}
	}
}
