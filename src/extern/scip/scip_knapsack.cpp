#include <vector>
#include "highs/util/HighsIntegers.h"
#include "utils.h"
#include "scip_knapsack.h"
#include <climits>

/* swapping two variables */
#define SORTTPL_SWAP(T,x,y) \
   {                \
      T temp = x;   \
      x = y;        \
      y = temp;     \
   }


/** macro that performs an exchange in the weighted selection algorithm, including weights */
#define EXCH(x,y)                                                                              \
   do                                                                                          \
   {                                                                                           \
      SORTTPL_SWAP(double, key[x], key[y]);                                                    \
      SORTTPL_SWAP(double, weights[x], weights[y]);                                            \
      SORTTPL_SWAP(int, field1[x], field1[y]);                                              \
      SORTTPL_SWAP(double, field2[x], field2[y]);                                              \
      SORTTPL_SWAP(int, field3[x], field3[y]);                                              \
   }                                                                                           \
   while( 0 )

/** returns the index a, b, or c of the median element among key[a], key[b], and key[c] */
/* let the elements in the unsorted order be a, b, c at positions start, mid, and end */
int sorttpl_medianThree(double* key, int a, int b, int c) {
	if (SORTTPL_ISBETTER(key[a], key[b])) {
		return SORTTPL_ISBETTER(key[b], key[c]) ? b : (SORTTPL_ISBETTER(key[a], key[c]) ? c : a);
	}
	else {
		return SORTTPL_ISBETTER(key[b], key[c]) ? (SORTTPL_ISBETTER(key[a], key[c]) ? a : c) : b;
	}
}

/** guess a median for the key array [start, ..., end] by using the median of the first, last, and middle element */
int sorttpl_selectPivotIndex(
	double* key,                /**< pointer to data array that defines the order */
	int                   start,              /**< first index of the key array to consider */
	int                   end                 /**< last index of the key array to consider */
)
{
	int pivotindex;

	/* use the middle index on small arrays */
	if (end - start + 1 <= SORTTPL_SHELLSORTMAX)
		pivotindex = (start + end) / 2;
	else if (end - start + 1 < SORTTPL_MINSIZENINTHER)
	{
		/* select the median of the first, last, and middle element as pivot element */
		int mid = (start + end) / 2;
		pivotindex = sorttpl_medianThree(key, start, mid, end);
	}
	else
	{
		/* use the median of medians of nine evenly distributed elements of the key array */
		int gap = (end - start + 1) / 9;
		int median1;
		int median2;
		int median3;

		/* collect 3 medians evenly distributed over the array */
		median1 = sorttpl_medianThree(key, start, start + gap, start + 2 * gap);
		median2 = sorttpl_medianThree(key, start + 3 * gap, start + 4 * gap, start + 5 * gap);
		median3 = sorttpl_medianThree(key, start + 6 * gap, start + 7 * gap, start + 8 * gap);

		/* compute and return the median of the medians */
		pivotindex = sorttpl_medianThree(key, median1, median2, median3);
	}

	return pivotindex;
}

void SCIPselectWeightedDownRealLongRealInt
(
	double* key,
	int* field1,
	double* field2,
	int* field3,
	double* weights, /**< (optional), nonnegative weights array for weighted median, or NULL (all weights are equal to 1) */
	double capacity, /**< the maximum capacity that is exceeded by the median */
	int len,         /**< length of arrays */
	int* medianpos   /**< pointer to store the index of the weighted median, or NULL, if not needed */
)
{
	int hi;
	int lo;
	int j;
	int localmedianpos = -1;
	double totalweightsum = 0.0;
	double residualcapacity;
	int itcount = 0;

	lo = 0;
	hi = len - 1;
	residualcapacity = capacity;

	/* compute the total weight and stop if all items fit */
	for (j = 0; j < len; ++j)
		totalweightsum += weights[j];

	if (totalweightsum <= capacity)
	{
		localmedianpos = len;
		goto CHECKANDRETURN;
	}

	while (hi - lo + 1 > SORTTPL_SHELLSORTMAX)
	{
		++itcount;
		int i;
		int bt;
		int wt;
		int p;
		int pivotindex;
		double betterweightsum;
		double pivotweight;
		double pivot;

		/* guess a median as pivot */
		pivotindex = sorttpl_selectPivotIndex(key, lo, hi);

		pivot = key[pivotindex];

		/* swap pivot element to the end of the array */
		if (pivotindex != lo)
		{
			EXCH(lo, pivotindex);
		}

		/* initialize array indices for the current element, the better elements, and the worse elements */
		i = lo;
		bt = lo;
		wt = hi;

		/* iterate through elements once to establish three partition into better elements, equal elements, and worse elements
		 *
		 * at every iteration, i denotes the current, previously unseen element, starting from the position lo
		 * all elements [lo,...bt - 1] are better than the pivot
		 * all elements [wt + 1,... hi] are worse than the pivot
		 *
		 * at termination, all elements [bt,...wt] are equal to the pivot element
		 * */
		while (i <= wt)
		{
			/* element i is better than pivot; exchange elements i and bt, increase both */
			if (SORTTPL_ISBETTER(key[i], pivot))
			{
				EXCH(i, bt);
				i++;
				bt++;
			}
			/* element i is worse than pivot: exchange it with the element at position wt; no increment of i
			 * because an unseen element is waiting at index i after the swap
			 */
			else if (SORTTPL_ISWORSE(key[i], pivot))
			{
				EXCH(i, wt);
				wt--;
			}
			else
				i++;
		}

		/* collect weights of elements larger than the pivot  */
		betterweightsum = 0.0;
		for (i = lo; i < bt; ++i)
		{
			betterweightsum += weights[i];
		}

		/* the weight in the better half of the array exceeds the capacity. Continue the search there */
		if (betterweightsum > residualcapacity)
		{
			hi = bt - 1;
		}
		else
		{
			double weightsum = betterweightsum;

			/* loop through duplicates of pivot element and check if one is the weighted median */
			for (p = bt; p <= wt; ++p)
			{
				pivotweight = weights[p];
				weightsum += pivotweight;

				/* the element at index p is exactly the weighted median */
				if (weightsum > residualcapacity)
				{
					localmedianpos = p;
					goto CHECKANDRETURN;
				}
			}

			/* continue loop by searching the remaining elements [wt+1,...,hi] */
			residualcapacity -= weightsum;
			lo = wt + 1;
		}
	}

	/* use shell sort to solve the remaining elements completely */
	if (hi - lo + 1 > 1)
	{
		sorttpl_shellSort(key, lo, hi, weights, field1, field2, field3);
	}

	/* it is impossible for lo or high to reach the end of the array. In this case, the item weights sum up to
	 * less than the capacity, which is handled at the top of this method.
	 */
	 /* determine the median position among the remaining elements */
	for (j = lo; j <= std::max(lo, hi); ++j)
	{
		double weight = weights[j];

		/* we finally found the median element */
		if (weight > residualcapacity)
		{
			localmedianpos = j;
			break;
		}
		else
			residualcapacity -= weight;
	}

CHECKANDRETURN:

	if (medianpos != NULL)
		*medianpos = localmedianpos;

	return;
}

int SCIPsolveKnapsackApproximately(
	int                   nitems,             /**< number of available items */
	int* weights,            /**< item weights */
	double* profits,            /**< item profits */
	long          capacity,           /**< capacity of knapsack */
	int* items,              /**< item numbers */
	int* solitems,           /**< array to store items in solution, or NULL */
	int* nonsolitems,        /**< array to store items not in solution, or NULL */
	int* nsolitems,          /**< pointer to store number of items in solution, or NULL */
	int* nnonsolitems,       /**< pointer to store number of items not in solution, or NULL */
	double* solval              /**< pointer to store optimal solution value, or NULL */
)
{
	long solitemsweight;
	int j;
	int criticalindex;

	if (solitems != NULL)
	{
		*nsolitems = 0;
		*nnonsolitems = 0;
	}
	if (solval != NULL)
		*solval = 0.0;

	/* initialize data for median search */
	std::vector<double> tempsort;
	tempsort.resize(nitems);

	std::vector<double> realweights;
	realweights.resize(nitems);

	for (j = nitems - 1; j >= 0; --j)
	{
		tempsort[j] = profits[j] / ((double)weights[j]);
		realweights[j] = (double)weights[j];
	}

	/* partially sort indices such that all elements that are larger than the break item appear first */
	SCIPselectWeightedDownRealLongRealInt(tempsort.data(), weights, profits, items, realweights.data(), (double)capacity, nitems, &criticalindex);

	/* selects items as long as they fit into the knapsack */
	solitemsweight = 0;
	for (j = 0; j < nitems && solitemsweight + weights[j] <= capacity; ++j)
	{
		if (solitems != NULL)
			solitems[(*nsolitems)++] = items[j];

		if (solval != NULL)
			(*solval) += profits[j];
		solitemsweight += weights[j];
	}

	if (solval != NULL)
		(*solval) += (capacity - solitemsweight) * tempsort[j];

	if (solitems != NULL)
	{
		for (; j < nitems; j++)
			nonsolitems[(*nnonsolitems)++] = items[j];
	}

	return 0;
}

/** quick-sort an array of pointers; pivot is the medial element */
static void sorttpl_qSort
(
	double* key,      /**< pointer to data array that defines the order */
	int* field1,      /**< additional field that should be sorted in the same way */
	int* field2,      /**< additional field that should be sorted in the same way */
	int start,        /**< starting index */
	int end,          /**< ending index */
	bool type         /**< TRUE, if quick-sort should start with with key[lo] < pivot <= key[hi], key[lo] <= pivot < key[hi] otherwise */
)
{
	assert(start <= end);

	/* use quick-sort for long lists */
	while (end - start >= SORTTPL_SHELLSORTMAX)
	{
		double pivotkey;
		int lo;
		int hi;
		int mid;

		/* select pivot element */
		mid = sorttpl_selectPivotIndex(key, start, end);
		pivotkey = key[mid];

		/* partition the array into elements < pivot [start,hi] and elements >= pivot [lo,end] */
		lo = start;
		hi = end;
		for (;; )
		{
			if (type)
			{
				while (lo < end && SORTTPL_ISBETTER(key[lo], pivotkey))
					lo++;
				while (hi > start && !SORTTPL_ISBETTER(key[hi], pivotkey))
					hi--;
			}
			else
			{
				while (lo < end && !SORTTPL_ISWORSE(key[lo], pivotkey))
					lo++;
				while (hi > start && SORTTPL_ISWORSE(key[hi], pivotkey))
					hi--;
			}

			if (lo >= hi)
				break;

			SORTTPL_SWAP(double, key[lo], key[hi]);
			SORTTPL_SWAP(int, field1[lo], field1[hi]);
			SORTTPL_SWAP(int, field2[lo], field2[hi]);
			lo++;
			hi--;
		}
		assert((hi == lo - 1) || (type && hi == start) || (!type && lo == end));

		/* skip entries which are equal to the pivot element (three partitions, <, =, > than pivot)*/
		if (type)
		{
			while (lo < end && !SORTTPL_ISBETTER(pivotkey, key[lo]))
				lo++;

			/* make sure that we have at least one element in the smaller partition */
			if (lo == start)
			{
				/* everything is greater or equal than the pivot element: move pivot to the left (degenerate case) */
				assert(!SORTTPL_ISBETTER(key[mid], pivotkey)); /* the pivot element did not change its position */
				assert(!SORTTPL_ISBETTER(pivotkey, key[mid]));
				SORTTPL_SWAP(double, key[lo], key[mid]);
				SORTTPL_SWAP(int, field1[lo], field1[mid]);
				SORTTPL_SWAP(int, field2[lo], field2[mid]);
				lo++;
			}
		}
		else
		{
			while (hi > start && !SORTTPL_ISWORSE(pivotkey, key[hi]))
				hi--;

			/* make sure that we have at least one element in the smaller partition */
			if (hi == end)
			{
				/* everything is greater or equal than the pivot element: move pivot to the left (degenerate case) */
				assert(!SORTTPL_ISBETTER(key[mid], pivotkey)); /* the pivot element did not change its position */
				assert(!SORTTPL_ISBETTER(pivotkey, key[mid]));
				SORTTPL_SWAP(double, key[hi], key[mid]);
				SORTTPL_SWAP(int, field1[hi], field1[mid]);
				SORTTPL_SWAP(int, field2[hi], field2[mid]);
				hi--;
			}
		}

		/* sort the smaller partition by a recursive call, sort the larger part without recursion */
		if (hi - start <= end - lo)
		{
			/* sort [start,hi] with a recursive call */
			if (start < hi)
			{
				sorttpl_qSort(key, field1, field2, start, hi, !type);
			}

			/* now focus on the larger part [lo,end] */
			start = lo;
		}
		else
		{
			if (lo < end)
			{
				/* sort [lo,end] with a recursive call */
				sorttpl_qSort(key, field1, field2, lo, end, !type);
			}

			/* now focus on the larger part [start,hi] */
			end = hi;
		}
		type = !type;
	}

	/* use shell sort on the remaining small list */
	if (end - start >= 1)
	{
		sorttpl_shellSort(key, start, end, field1, field2);
	}
}

/** SCIPsort...(): sorts array 'key' and performs the same permutations on the additional 'field' arrays */
void SCIPsortDownRealIntLong (
	double* key,  /**< pointer to data array that defines the order */
	int* field1,   /**< additional field that should be sorted in the same way */
	int* field2,  /**< additional field that should be sorted in the same way */
	int len /**< length of arrays */
	)
{
	/* ignore the trivial cases */
	if (len <= 1)
		return;

	/* use shell sort on the remaining small list */
	if (len <= SORTTPL_SHELLSORTMAX) {
		sorttpl_shellSort(key, 0, len - 1, field1, field2);
	}
	else {
		sorttpl_qSort(key, field1, field2, 0, len - 1, true);
	}
}

void SCIPsolveKnapsackExactly(
	int nitems,             /**< number of available items */
	int* weights,            /**< item weights */
	double* profits,            /**< item profits */
	uint32_t capacity,           /**< capacity of knapsack */
	int* items,              /**< item numbers */
	int* solitems,           /**< array to store items in solution, or NULL */
	int* nonsolitems,        /**< array to store items not in solution, or NULL */
	int* nsolitems,          /**< pointer to store number of items in solution, or NULL */
	int* nnonsolitems,       /**< pointer to store number of items not in solution, or NULL */
	double* solval,             /**< pointer to store optimal solution value, or NULL */
	bool* success             /**< pointer to store if an error occured during solving (normally a memory problem) */
)
{
	int intcap;
	int d;
	int j;
	int greedymedianpos;
	uint32_t weightsum;
	int nmyitems;
	uint32_t gcd;
	uint32_t minweight;
	uint32_t maxweight;
	int currminweight;
	uint32_t greedysolweight;
	double greedysolvalue;
	double greedyupperbound;
	bool eqweights;
	bool intprofits;

	std::vector<int> myitems(nitems);
	std::vector<int> myweights(nitems);
	std::vector<double> myprofits(nitems);

	std::vector<int> allcurrminweight;
	std::vector<double> optvalues;
	std::vector<double> tempsort;
	std::vector<double> realweights;

	*success = true;

	/* initializing solution value */
	if (solval != NULL)
		*solval = 0.0;

	/* init solution information */
	if (solitems != NULL)
	{
		*nnonsolitems = 0;
		*nsolitems = 0;
	}

	/* allocate temporary memory */
	nmyitems = 0;
	weightsum = 0;
	minweight = std::numeric_limits<uint32_t>::max();
	maxweight = 0;

	/* remove unnecessary items */
	for (j = 0; j < nitems; ++j)
	{
		/* item does not fit */
		if (weights[j] > capacity)
		{
			if (solitems != NULL)
				nonsolitems[(*nnonsolitems)++] = items[j]; /*lint !e413*/
		}
		/* item is not profitable */
		else if (profits[j] <= 0.0)
		{
			if (solitems != NULL)
				nonsolitems[(*nnonsolitems)++] = items[j]; /*lint !e413*/
		}
		/* item always fits */
		else if (weights[j] == 0)
		{
			if (solitems != NULL)
				solitems[(*nsolitems)++] = items[j]; /*lint !e413*/

			if (solval != NULL)
				*solval += profits[j];
		}
		/* all important items */
		else
		{
			myweights[nmyitems] = weights[j];
			myprofits[nmyitems] = profits[j];
			myitems[nmyitems] = items[j];

			/* remember smallest item */
			if (myweights[nmyitems] < minweight)
				minweight = myweights[nmyitems];

			/* remember bigest item */
			if (myweights[nmyitems] > maxweight)
				maxweight = myweights[nmyitems];

			weightsum += myweights[nmyitems];
			++nmyitems;
		}
	}

	intprofits = true;
	/* check if all profits are integer to strengthen the upper bound on the greedy solution */
	for (j = 0; j < nmyitems && intprofits; ++j)
		intprofits = intprofits && HighsIntegers::isIntegral(myprofits[j], 1e-6);

	/* if no item is left then goto end */
	if (nmyitems == 0)
	{
		//SCIPdebugMsg(scip, "After preprocessing no items are left.\n");
		goto TERMINATE;
	}

	/* if all items fit, we also do not need to do the expensive stuff later on */
	if (weightsum > 0 && weightsum <= capacity)
	{
		//SCIPdebugMsg(scip, "After preprocessing all items fit into knapsack.\n");

		for (j = nmyitems - 1; j >= 0; --j)
		{
			if (solitems != NULL)
				solitems[(*nsolitems)++] = myitems[j]; /*lint !e413*/

			if (solval != NULL)
				*solval += myprofits[j];
		}

		goto TERMINATE;
	}

	/* make weights relatively prime */
	eqweights = true;
	if (maxweight > 1)
	{
		/* determine greatest common divisor */
		gcd = myweights[nmyitems - 1];
		for (j = nmyitems - 2; j >= 0 && gcd >= 2; --j)
			gcd = calc_gcd(gcd, myweights[j]);

		/* divide by greatest common divisor */
		if (gcd > 1)
		{
			for (j = nmyitems - 1; j >= 0; --j)
			{
				myweights[j] /= gcd;
				eqweights = eqweights && (myweights[j] == 1);
			}
			capacity /= gcd;
			minweight /= gcd;
		}
		else
			eqweights = false;
	}

	/* if only one item fits, then take the best */
	if (minweight > capacity / 2)
	{
		int p;
		//SCIPdebugMsg(scip, "Only one item fits into knapsack, so take the best.\n");

		p = nmyitems - 1;

		/* find best item */
		for (j = nmyitems - 2; j >= 0; --j)
		{
			if (myprofits[j] > myprofits[p])
				p = j;
		}

		/* update solution information */
		if (solitems != NULL)
		{
			assert(nsolitems != NULL && nonsolitems != NULL && nnonsolitems != NULL);

			solitems[(*nsolitems)++] = myitems[p];
			for (j = nmyitems - 1; j >= 0; --j)
			{
				if (j != p)
					nonsolitems[(*nnonsolitems)++] = myitems[j];
			}
		}
		/* update solution value */
		if (solval != NULL)
			*solval += myprofits[p];

		goto TERMINATE;
	}

	/* if all items have the same weight, then take the best */
	if (eqweights)
	{
		double addval = 0.0;
		//SCIPdebugMsg(scip, "All weights are equal, so take the best.\n");
		std::cout << "Unsupported operation" << std::endl;
		exit(0);

		SCIPsortDownRealIntLong(myprofits.data(), myitems.data(), myweights.data(), nmyitems);

		/* update solution information */
		if (solitems != NULL || solval != NULL)
		{
		    int64_t i;

		    /* take the first best items into the solution */
		    for (i = capacity - 1; i >= 0; --i)
		    {
		        if (solitems != NULL)
		            solitems[(*nsolitems)++] = myitems[i];
		        addval += myprofits[i];
		    }

		    if (solitems != NULL)
		    {
		        /* the rest are not in the solution */
		        for (i = nmyitems - 1; i >= capacity; --i)
		            nonsolitems[(*nnonsolitems)++] = myitems[i];
		    }
		}
		/* update solution value */
		if (solval != NULL)
		{
		    assert(addval > 0.0);
		    *solval += addval;
		}

		goto TERMINATE;
	}


	/* sort myitems (plus corresponding arrays myweights and myprofits) such that
	 * p_1/w_1 >= p_2/w_2 >= ... >= p_n/w_n, this is only used for the greedy solution
	 */
	tempsort.resize(nmyitems);
	realweights.resize(nmyitems);

	for (j = 0; j < nmyitems; ++j)
	{
		tempsort[j] = myprofits[j] / ((double)myweights[j]);
		realweights[j] = (double)myweights[j];
	}

	SCIPselectWeightedDownRealLongRealInt(tempsort.data(), myweights.data(), myprofits.data(), myitems.data(), realweights.data(),
		(double)capacity, nmyitems, &greedymedianpos);

	/* initialize values for greedy solution information */
	greedysolweight = 0;
	greedysolvalue = 0.0;

	/* determine greedy solution */
	for (j = 0; j < greedymedianpos; ++j)
	{
		assert(myweights[j] <= capacity);

		/* update greedy solution weight and value */
		greedysolweight += myweights[j];
		greedysolvalue += myprofits[j];
	}

	assert(0 < greedysolweight && greedysolweight <= capacity);
	assert(greedysolvalue > 0.0);

	/* If the greedy solution is optimal by comparing to the LP solution, we take this solution. This happens if:
	 * - the greedy solution reaches the capacity, because then the LP solution is integral;
	 * - the greedy solution has an objective that is at least the LP value rounded down in case that all profits are integer, too. */
	greedyupperbound = greedysolvalue + myprofits[j] * (double)(capacity - greedysolweight) / ((double)myweights[j]);
	if (intprofits)
		greedyupperbound = floor(greedyupperbound);
	if (greedysolweight == capacity || greedysolvalue >= greedyupperbound - 1e-9)
	{
		//SCIPdebugMsg(scip, "Greedy solution is optimal.\n");

		/* update solution information */
		if (solitems != NULL)
		{
			int l;

			assert(nsolitems != NULL && nonsolitems != NULL && nnonsolitems != NULL);

			/* collect items */
			for (l = 0; l < j; ++l)
				solitems[(*nsolitems)++] = myitems[l];
			for (; l < nmyitems; ++l)
				nonsolitems[(*nnonsolitems)++] = myitems[l];
		}
		/* update solution value */
		if (solval != NULL)
		{
			assert(greedysolvalue > 0.0);
			*solval += greedysolvalue;
		}

		goto TERMINATE;
	}

	/* in the following table we do not need the first minweight columns */
	capacity -= (minweight - 1);

	/* we can only handle integers */
	if (capacity >= INT_MAX)
	{
		//SCIPdebugMsg(scip, "Capacity is to big, so we cannot handle it here.\n");

		*success = false;
		goto TERMINATE;
	}
	assert(capacity < INT_MAX);

	intcap = (int)capacity;
	assert(intcap >= 0);
	assert(nmyitems > 0);
	assert(sizeof(size_t) >= sizeof(int)); /*lint !e506*/ /* no following conversion should be messed up */

	/* this condition checks whether we will try to allocate a correct number of bytes and do not have an overflow, while
	 * computing the size for the allocation
	 */
	if (intcap < 0 || (intcap > 0 && (((size_t)nmyitems) > (SIZE_MAX / (size_t)intcap / sizeof(double)) || ((size_t)nmyitems) * ((size_t)intcap) * sizeof(double) > ((size_t)INT_MAX)))) /*lint !e571*/
	{
		//SCIPdebugMsg(scip, "Too much memory (%lu) would be consumed.\n", (unsigned long)(((size_t)nmyitems) * ((size_t)intcap) * sizeof(*optvalues))); /*lint !e571*/

		*success = false;
		goto TERMINATE;
	}


	/* allocate temporary memory and check for memory exceedance */
	optvalues.resize(nmyitems * intcap);

	//retcode = SCIPallocBufferArray(scip, &optvalues, nmyitems * intcap);
	//if (retcode == SCIP_NOMEMORY)
	//{
	//    SCIPdebugMsg(scip, "Did not get enough memory.\n");

	//    *success = FALSE;
	//    goto TERMINATE;
	//}
	//else
	//{
	//    SCIP_CALL(retcode);
	//}

	//SCIPdebugMsg(scip, "Start real exact algorithm.\n");

	/* we memorize at each step the current minimal weight to later on know which value in our optvalues matrix is valid;
	 * each value entries of the j-th row of optvalues is valid if the index is >= allcurrminweight[j], otherwise it is
	 * invalid; a second possibility would be to clear the whole optvalues, which should be more expensive than storing
	 * 'nmyitem' values
	 */
	allcurrminweight.resize(nmyitems);
	assert(myweights[0] - minweight < INT_MAX);
	currminweight = (int)(myweights[0] - minweight);
	allcurrminweight[0] = currminweight;

	/* fills first row of dynamic programming table with optimal values */
	for (d = currminweight; d < intcap; ++d)
		optvalues[d] = myprofits[0];

#define IDX(j,d) ((j)*(intcap)+(d))

	/* fills dynamic programming table with optimal values */
	for (j = 1; j < nmyitems; ++j)
	{
		int intweight;

		/* compute important part of weight, which will be represented in the table */
		intweight = (int)(myweights[j] - minweight);
		assert(0 <= intweight && intweight < intcap);

		/* copy all nonzeros from row above */
		for (d = currminweight; d < intweight && d < intcap; ++d)
			optvalues[IDX(j, d)] = optvalues[IDX(j - 1, d)];

		/* update corresponding row */
		for (d = intweight; d < intcap; ++d)
		{
			/* if index d < current minweight then optvalues[IDX(j-1,d)] is not initialized, i.e. should be 0 */
			if (d < currminweight)
				optvalues[IDX(j, d)] = myprofits[j];
			else
			{
				double sumprofit;

				if (d - myweights[j] < currminweight)
					sumprofit = myprofits[j];
				else
					sumprofit = optvalues[IDX(j - 1, (int)(d - myweights[j]))] + myprofits[j];

				optvalues[IDX(j, d)] = std::max(sumprofit, optvalues[IDX(j - 1, d)]);
			}
		}

		/* update currminweight */
		if (intweight < currminweight)
			currminweight = intweight;

		allcurrminweight[j] = currminweight;
	}

	/* update optimal solution by following the table */
	if (solitems != NULL)
	{
		assert(nsolitems != NULL && nonsolitems != NULL && nnonsolitems != NULL);
		d = intcap - 1;

		//SCIPdebugMsg(scip, "Fill the solution vector after solving exactly.\n");

		/* insert all items in (non-) solution vector */
		for (j = nmyitems - 1; j > 0; --j)
		{
			/* if the following condition holds this means all remaining items does not fit anymore */
			if (d < allcurrminweight[j])
			{
				/* we cannot have exceeded our capacity */
				break;
			}

			/* collect solution items; the first condition means that no further item can fit anymore, but this does */
			if (d < allcurrminweight[j - 1] || optvalues[IDX(j, d)] > optvalues[IDX(j - 1, d)])
			{
				solitems[(*nsolitems)++] = myitems[j];

				/* check that we do not have an underflow */
				d = (int)(d - myweights[j]);
			}
			/* collect non-solution items */
			else
				nonsolitems[(*nnonsolitems)++] = myitems[j];
		}

		/* insert remaining items */
		if (d >= allcurrminweight[j])
		{
			assert(j == 0);
			solitems[(*nsolitems)++] = myitems[j];
		}
		else
		{
			assert(j >= 0);
			assert(d < allcurrminweight[j]);

			for (; j >= 0; --j)
				nonsolitems[(*nnonsolitems)++] = myitems[j];
		}

		assert(*nsolitems + *nnonsolitems == nitems);
	}

	/* update solution value */
	if (solval != NULL)
		*solval += optvalues[IDX(nmyitems - 1, intcap - 1)];

TERMINATE:
	return;
}


bool SCIPsolveKnapsackExactly(
	int nitems,        /**< number of available items */
	const int* weights,      /**< item weights */
	const double* profits,   /**< item profits */
	uint32_t capacity, /**< capacity of knapsack */
	int* solitems,     /**< array to store items in solution */
	int* nsolitems,    /**< pointer to store number of items in solution */
	double* solval    /**< pointer to store optimal solution value */
)
{
	int intcap;
	int d;
	int j;
	int greedymedianpos;
	uint32_t weightsum;
	int nmyitems;
	uint32_t gcd;
	uint32_t minweight;
	uint32_t maxweight;
	int currminweight;
	uint32_t greedysolweight;
	double greedysolvalue;
	double greedyupperbound;
	bool eqweights;
	bool intprofits;
	bool success = true;

	std::vector<int> myitems(nitems);
	std::vector<int> myweights(nitems);
	std::vector<double> myprofits(nitems);

	std::vector<int> allcurrminweight;
	std::vector<double> optvalues;
	std::vector<double> tempsort;
	std::vector<double> realweights;

	*solval = 0.0;
	*nsolitems = 0;

	/* allocate temporary memory */
	nmyitems = 0;
	weightsum = 0;
	minweight = std::numeric_limits<uint32_t>::max();
	maxweight = 0;

	/* remove unnecessary items */
	for (j = 0; j < nitems; ++j)
	{
		/* item does not fit or item is not profitable */
		if (weights[j] > capacity || profits[j] <= 0.0) {
			/* ignore */
		}
		/* item always fits */
		else if (weights[j] == 0) {
			solitems[(*nsolitems)++] = j; /*lint !e413*/
			*solval += profits[j];
		}
		/* all important items */
		else {
			myweights[nmyitems] = weights[j];
			myprofits[nmyitems] = profits[j];
			myitems[nmyitems] = j;

			/* remember smallest item */
			if (myweights[nmyitems] < minweight)
				minweight = myweights[nmyitems];

			/* remember bigest item */
			if (myweights[nmyitems] > maxweight)
				maxweight = myweights[nmyitems];

			weightsum += myweights[nmyitems];
			++nmyitems;
		}
	}

	intprofits = true;
	/* check if all profits are integer to strengthen the upper bound on the greedy solution */
	for (j = 0; j < nmyitems && intprofits; ++j)
		intprofits = intprofits && HighsIntegers::isIntegral(myprofits[j], 1e-6);

	/* if no item is left then goto end */
	if (nmyitems == 0) {
		goto TERMINATE;
	}

	/* if all items fit, we also do not need to do the expensive stuff later on */
	if (weightsum > 0 && weightsum <= capacity)	{
		for (j = nmyitems - 1; j >= 0; --j) {
			solitems[(*nsolitems)++] = myitems[j]; /*lint !e413*/
			*solval += myprofits[j];
		}

		goto TERMINATE;
	}

	/* make weights relatively prime */
	eqweights = true;
	if (maxweight > 1) {
		/* determine greatest common divisor */
		gcd = myweights[nmyitems - 1];
		for (j = nmyitems - 2; j >= 0 && gcd >= 2; --j)
			gcd = calc_gcd(gcd, myweights[j]);

		/* divide by greatest common divisor */
		if (gcd > 1) {
			for (j = nmyitems - 1; j >= 0; --j) {
				myweights[j] /= gcd;
				eqweights = eqweights && (myweights[j] == 1);
			}
			capacity /= gcd;
			minweight /= gcd;
		}
		else
			eqweights = false;
	}

	/* if only one item fits, then take the best */
	if (minweight > capacity / 2) {
		int p;
		p = nmyitems - 1;

		/* find best item */
		for (j = nmyitems - 2; j >= 0; --j) {
			if (myprofits[j] > myprofits[p])
				p = j;
		}

		/* update solution information */
		assert(nsolitems != NULL);
		solitems[(*nsolitems)++] = myitems[p];

		*solval += myprofits[p];
		goto TERMINATE;
	}

	/* if all items have the same weight, then take the best */
	if (eqweights) {
		double addval = 0.0;
		SCIPsortDownRealIntLong(myprofits.data(), myitems.data(), myweights.data(), nmyitems);

		/* take the first best items into the solution */
		for (int i = static_cast<int>(capacity) - 1; i >= 0; --i) {
			solitems[(*nsolitems)++] = myitems[i];
			addval += myprofits[i];
		}

		assert(addval > 0.0);
		*solval += addval;
		goto TERMINATE;
	}


	// sort myitems (plus corresponding arrays myweights and myprofits) such that
	// p_1/w_1 >= p_2/w_2 >= ... >= p_n/w_n, this is only used for the greedy solution
	tempsort.resize(nmyitems);
	realweights.resize(nmyitems);

	for (j = 0; j < nmyitems; ++j) {
		tempsort[j] = myprofits[j] / ((double)myweights[j]);
		realweights[j] = (double)myweights[j];
	}

	SCIPselectWeightedDownRealLongRealInt(tempsort.data(), myweights.data(), myprofits.data(), myitems.data(), realweights.data(),
		(double)capacity, nmyitems, &greedymedianpos);

	/* initialize values for greedy solution information */
	greedysolweight = 0;
	greedysolvalue = 0.0;

	/* determine greedy solution */
	for (j = 0; j < greedymedianpos; ++j) {
		assert(myweights[j] <= capacity);

		/* update greedy solution weight and value */
		greedysolweight += myweights[j];
		greedysolvalue += myprofits[j];
	}

	assert(0 < greedysolweight && greedysolweight <= capacity);
	assert(greedysolvalue > 0.0);

	/* If the greedy solution is optimal by comparing to the LP solution, we take this solution. This happens if:
	 * - the greedy solution reaches the capacity, because then the LP solution is integral;
	 * - the greedy solution has an objective that is at least the LP value rounded down in case that all profits are integer, too. */
	greedyupperbound = greedysolvalue + myprofits[j] * (double)(capacity - greedysolweight) / ((double)myweights[j]);
	if (intprofits)
		greedyupperbound = floor(greedyupperbound);

	// Greedy solution is optimal
	if (greedysolweight == capacity || greedysolvalue >= greedyupperbound - 1e-9) {
		int l;
		assert(nsolitems != NULL);

		for (l = 0; l < j; ++l)
			solitems[(*nsolitems)++] = myitems[l];

		assert(greedysolvalue > 0.0);
		*solval += greedysolvalue;
		goto TERMINATE;
	}

	/* in the following table we do not need the first minweight columns */
	capacity -= (minweight - 1);

	/* we can only handle integers */
	// Capacity is to big, so we cannot handle it here.
	if (capacity >= INT_MAX) {
		success = false;
		goto TERMINATE;
	}

	assert(capacity < INT_MAX);
	intcap = (int)capacity;
	assert(intcap >= 0);
	assert(nmyitems > 0);
	assert(sizeof(size_t) >= sizeof(int)); /*lint !e506*/ /* no following conversion should be messed up */

	/* this condition checks whether we will try to allocate a correct number of bytes and do not have an overflow, while
	 * computing the size for the allocation
	 */
	if (intcap < 0 || (intcap > 0 && (((size_t)nmyitems) > (SIZE_MAX / (size_t)intcap / sizeof(double)) || ((size_t)nmyitems) * ((size_t)intcap) * sizeof(double) > ((size_t)INT_MAX)))) /*lint !e571*/
	{
		//SCIPdebugMsg(scip, "Too much memory (%lu) would be consumed.\n", (unsigned long)(((size_t)nmyitems) * ((size_t)intcap) * sizeof(*optvalues))); /*lint !e571*/
		success = false;
		goto TERMINATE;
	}

	/* allocate temporary memory and check for memory exceedance */
	optvalues.resize(nmyitems * intcap);

	/* we memorize at each step the current minimal weight to later on know which value in our optvalues matrix is valid;
	 * each value entries of the j-th row of optvalues is valid if the index is >= allcurrminweight[j], otherwise it is
	 * invalid; a second possibility would be to clear the whole optvalues, which should be more expensive than storing
	 * 'nmyitem' values
	 */
	allcurrminweight.resize(nmyitems);
	assert(myweights[0] - minweight < INT_MAX);
	currminweight = (int)(myweights[0] - minweight);
	allcurrminweight[0] = currminweight;

	/* fills first row of dynamic programming table with optimal values */
	for (d = currminweight; d < intcap; ++d)
		optvalues[d] = myprofits[0];

#define IDX(j,d) ((j)*(intcap)+(d))

	/* fills dynamic programming table with optimal values */
	for (j = 1; j < nmyitems; ++j) {
		int intweight;

		/* compute important part of weight, which will be represented in the table */
		intweight = (int)(myweights[j] - minweight);
		assert(0 <= intweight && intweight < intcap);

		/* copy all nonzeros from row above */
		for (d = currminweight; d < intweight && d < intcap; ++d)
			optvalues[IDX(j, d)] = optvalues[IDX(j - 1, d)];

		/* update corresponding row */
		for (d = intweight; d < intcap; ++d) {
			/* if index d < current minweight then optvalues[IDX(j-1,d)] is not initialized, i.e. should be 0 */
			if (d < currminweight)
				optvalues[IDX(j, d)] = myprofits[j];
			else
			{
				double sumprofit;

				if (d - myweights[j] < currminweight)
					sumprofit = myprofits[j];
				else
					sumprofit = optvalues[IDX(j - 1, (int)(d - myweights[j]))] + myprofits[j];

				optvalues[IDX(j, d)] = std::max(sumprofit, optvalues[IDX(j - 1, d)]);
			}
		}

		/* update currminweight */
		if (intweight < currminweight)
			currminweight = intweight;

		allcurrminweight[j] = currminweight;
	}

	/* update optimal solution by following the table */
	d = intcap - 1;

	/* insert all items in (non-) solution vector */
	for (j = nmyitems - 1; j > 0; --j) {
		/* if the following condition holds this means all remaining items does not fit anymore */
		if (d < allcurrminweight[j]) {
			/* we cannot have exceeded our capacity */
			break;
		}

		/* collect solution items; the first condition means that no further item can fit anymore, but this does */
		if (d < allcurrminweight[j - 1] || optvalues[IDX(j, d)] > optvalues[IDX(j - 1, d)])
		{
			solitems[(*nsolitems)++] = myitems[j];

			/* check that we do not have an underflow */
			d = (int)(d - myweights[j]);
		}
	}

	/* insert remaining items */
	if (d >= allcurrminweight[j])
	{
		assert(j == 0);
		solitems[(*nsolitems)++] = myitems[j];
	}

	*solval += optvalues[IDX(nmyitems - 1, intcap - 1)];

TERMINATE:
	return success;
}


