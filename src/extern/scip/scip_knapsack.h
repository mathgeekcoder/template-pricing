#pragma once
// This code was originally part of the SCIP Optimization Suite.
// It has been modified to make better usage of C++, and only includes
// the relevant parts needed for the knapsack solver.
// 
// Since this is derivative work, it is licensed as follows:
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
// http ://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define SORTTPL_SHELLSORTMAX    25 /* maximal size for shell sort */
#define SORTTPL_MINSIZENINTHER 729 /* minimum input size to use ninther (median of nine) for pivot selection */

#define SORTTPL_CMP(x,y) ((y) - (x))
#define SORTTPL_ISBETTER(x,y) (SORTTPL_CMP(x,y) < 0)
#define SORTTPL_ISWORSE(x,y) (SORTTPL_CMP(x,y) > 0)

template <typename T, typename... Args>
void sorttpl_shellSort(T* key, // pointer to data array that defines the order 
    int start, int end,        // start/end index
    Args*... fields)           // additional fields that should be permuted like key
{
    static const int incs[3] = { 1, 5, 19 }; // sequence of increments

    for (int k = 2; k >= 0; --k) {
        int h = incs[k];
        int first = h + start;

        for (int i = first; i <= end; ++i) {
            auto tempkey = key[i];
            auto tempfields = std::make_tuple(fields[i]...);

            int j = i;
            while (j >= first && SORTTPL_ISBETTER(tempkey, key[j - h])) {
                key[j] = key[j - h];
				std::tie(fields[j]...) = std::make_tuple(fields[j - h]...);
                j -= h;
            }

            key[j] = tempkey;
			std::tie(fields[j]...) = tempfields;
        }
    }
}

int SCIPsolveKnapsackApproximately(
	int nitems,              /**< number of available items */
	int* weights,            /**< item weights */
	double* profits,         /**< item profits */
	long capacity,           /**< capacity of knapsack */
	int* items,              /**< item numbers */
	int* solitems,           /**< array to store items in solution, or NULL */
	int* nonsolitems,        /**< array to store items not in solution, or NULL */
	int* nsolitems,          /**< pointer to store number of items in solution, or NULL */
	int* nnonsolitems,       /**< pointer to store number of items not in solution, or NULL */
	double* solval           /**< pointer to store optimal solution value, or NULL */
);

void SCIPsolveKnapsackExactly(
    int nitems,             /**< number of available items */
    int* weights,           /**< item weights */
    double* profits,        /**< item profits */
    uint32_t capacity,      /**< capacity of knapsack */
    int* items,             /**< item numbers */
    int* solitems,          /**< array to store items in solution, or NULL */
    int* nonsolitems,       /**< array to store items not in solution, or NULL */
    int* nsolitems,         /**< pointer to store number of items in solution, or NULL */
    int* nnonsolitems,      /**< pointer to store number of items not in solution, or NULL */
    double* solval,         /**< pointer to store optimal solution value, or NULL */
    bool* success           /**< pointer to store if an error occured during solving (normally a memory problem) */
);

bool SCIPsolveKnapsackExactly(
    int nitems,             /**< number of available items */
    const int* weights,     /**< item weights */
    const double* profits,  /**< item profits */
    uint32_t capacity,      /**< capacity of knapsack */
    int* solitems,          /**< array to store items in solution */
    int* nsolitems,         /**< pointer to store number of items in solution */
    double* solval,          /**< pointer to store optimal solution value */
    std::vector<double>& optvalues
);

static inline bool SCIPsolveKnapsackExactly(
    int nitems,             /**< number of available items */
    const int* weights,     /**< item weights */
    const double* profits,  /**< item profits */
    uint32_t capacity,      /**< capacity of knapsack */
    int* solitems,          /**< array to store items in solution */
    int* nsolitems,         /**< pointer to store number of items in solution */
    double* solval          /**< pointer to store optimal solution value */
) {
    std::vector<double> optvalues;
	return SCIPsolveKnapsackExactly(nitems, weights, profits, capacity, solitems, nsolitems, solval, optvalues);
}
