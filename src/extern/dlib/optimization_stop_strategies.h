// Copyright (C) 2008  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
//
// Modified by Luke Marshall to use eigen and extract relevant parts only
#pragma once

#include <cmath>
#include <limits>
#include <iostream>
#include <cassert>

namespace dlib 
{
	template <typename Func>
	class objective_delta_stop_strategy
	{
		Func* _callback;

	public:
		explicit objective_delta_stop_strategy(Func* callback = nullptr, double min_delta = 1e-7)
			: _callback(callback), _been_used(false), _min_delta(min_delta), _max_iter(0), _cur_iter(0), _prev_funct_value(0)
		{
			// min_delta can't be negative
			assert(min_delta >= 0);
		}

		objective_delta_stop_strategy(Func* callback, double min_delta, unsigned long max_iter)
			: _callback(callback), _been_used(false), _min_delta(min_delta), _max_iter(max_iter), _cur_iter(0), _prev_funct_value(0)
		{
			assert(min_delta >= 0 && max_iter > 0);
		}

		template <typename T>
		bool should_continue_search(const T&, const double funct_value, const T&)
		{
			if (_callback != nullptr && (*_callback)(_cur_iter, funct_value, std::abs(funct_value - _prev_funct_value)) == true)
				return false;

			++_cur_iter;
			if (_been_used) {
				// Check if we have hit the max allowable number of iterations.  (but only
				// check if _max_iter is enabled (i.e. not 0)).
				if (_max_iter != 0 && _cur_iter > _max_iter)
					return false;

				// check if the function change was too small
				if (std::abs(funct_value - _prev_funct_value) < _min_delta)
					return false;
			}

			_been_used = true;
			_prev_funct_value = funct_value;
			return true;
		}

	public:
		bool _been_used;
		double _min_delta;
		unsigned long _max_iter;
		unsigned long _cur_iter;
		double _prev_funct_value;
	};
}
