// Copyright (C) 2008  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
//
// Modified by Luke Marshall to use eigen and extract relevant parts only
#pragma once

#include <cmath>
#include <limits>
#include <cassert>
#include <Eigen/Core>
#include "sequence_kernel_2.h"
#include "optimization_line_search.h"

namespace dlib
{
	class lbfgs_search_strategy
	{
	public:
		explicit lbfgs_search_strategy(unsigned long max_size_) : max_size(max_size_), been_used(false)
		{
			assert(max_size > 0); // "lbfgs_search_strategy(max_size) max_size can't be zero"
		}

		lbfgs_search_strategy(const lbfgs_search_strategy& item)
		{
			max_size = item.max_size;
			been_used = item.been_used;
			prev_x = item.prev_x;
			prev_derivative = item.prev_derivative;
			prev_direction = item.prev_direction;
			alpha = item.alpha;
			dh_temp = item.dh_temp;
		}

		double get_wolfe_rho() const { return 0.01; }
		double get_wolfe_sigma() const { return 0.9; }
		unsigned long get_max_line_search_iterations() const { return 100; }

		template <typename T>
		const Eigen::VectorXd& get_next_direction(
			const T& x,
			const double,
			const T& funct_derivative
		)
		{
			prev_direction = -funct_derivative;

			if (been_used == false)
			{
				been_used = true;
			}
			else
			{
				// add an element into the stored data sequence
				dh_temp.s = x - prev_x;
				dh_temp.y = funct_derivative - prev_derivative;
				double temp = dh_temp.s.dot(dh_temp.y);
				// only accept this bit of data if temp isn't zero
				if (std::abs(temp) > std::numeric_limits<double>::epsilon())
				{
					dh_temp.rho = 1 / temp;
					data.add(data.size(), dh_temp);
				}
				else
				{
					data.clear();
				}

				if (data.size() > 0)
				{
					// This block of code is from algorithm 7.4 in the Nocedal book.

					alpha.resize(data.size());
					for (int i = data.size() - 1; i >= 0; --i) {
						alpha[i] = data[i].rho * data[i].s.dot(prev_direction);
						prev_direction -= alpha[i] * data[i].y;
					}

					// Take a guess at what the first H matrix should be.  This formula below is what is suggested
					// in the book Numerical Optimization by Nocedal and Wright in the chapter on Large Scale 
					// Unconstrained Optimization (in the L-BFGS section).
					const auto& last = data[data.size() - 1];
					double H_0 = 1.0 / last.rho / last.y.dot(last.y);
					H_0 = put_in_range(0.001, 1000.0, H_0);
					prev_direction *= H_0;

					for (unsigned long i = 0; i < data.size(); ++i) {
						double beta = data[i].rho * data[i].y.dot(prev_direction);
						prev_direction += data[i].s * (alpha[i] - beta);
					}
				}

			}

			if (data.size() > max_size)
			{
				// remove the oldest element in the data sequence
				data.remove(0, dh_temp);
			}

			prev_x = x;
			prev_derivative = funct_derivative;
			return prev_direction;
		}

	private:
		struct data_helper {
			Eigen::VectorXd s;
			Eigen::VectorXd y;
			double rho = 0.01;

			friend void swap(data_helper& a, data_helper& b)
			{
				a.s.swap(b.s);
				a.y.swap(b.y);
				std::swap(a.rho, b.rho);
			}
		};
		sequence_kernel_2<data_helper> data;

		unsigned long max_size;
		bool been_used;
		Eigen::VectorXd prev_x;
		Eigen::VectorXd prev_derivative;
		Eigen::VectorXd prev_direction;
		std::vector<double> alpha;

		data_helper dh_temp;
	};
}

