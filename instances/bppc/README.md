# Instances for Bin Packing Problem with Conflicts

Repository containing newly generated datasets for the Bin Packing Problem with Conflicts (BPPC) that were considered in the paper "Costa, Luciano; Contardo, Claudio; Desaulniers, Guy; Yarkony, Julian. Stabilized column generation via the dynamic separation of aggregated rows" accepted to be published in INFORMS Journal on Computing. 

These new instances, based on the traditional BPPC dataset introduced by Gendreau et al. (2004), contain arbitrary conflict graphs (ACGs) and were generated according to the procedure of Sadykov and Vanderbeck (2013). For more details on how the instances were generated, the interested reader is referred to the paper "Costa, Luciano; Contardo, Claudio; Desaulniers, Guy; Yarkony, Julian. Stabilized column generation via the dynamic separation of aggregated rows".

Each instance file is organized as follows:

n	W

1	w	a1	a2	...	ak

i	w	a1	a2	...	ak

n	W	a1	a2	...	ak

where:

* n is the number of items
* W is the bin capacity
* i is a number identifying the item
* w is the item weight
* a1, ..., ak are the items that are incompatible with the item i


### References

Gendreau, M; Laporte, G; Semet, F. (2004) Heuristics and lower bounds for the bin packing problem with conflicts. Computers & Operations Research 31(3):347–358.

Sadykov, R., Vanderbeck, F. (2013) Bin packing with conflicts: A generic branch-and-price algorithm. INFORMS Journal on Computing 25(2):244–255.

