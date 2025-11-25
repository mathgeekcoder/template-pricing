from __future__ import annotations
import argparse
from math import sqrt
import sys
from pathlib import Path
import polars as pl
import numpy as np
import pandas as pd
from statsmodels.stats.libqsturng import qsturng
import statsmodels.formula.api as smf
import statsmodels.api as sm

def hsu_mcb(data, groups, seeds, alpha=0.05, delta=0.01, delta_abs=0.5):
    """
    Perform Hsu's Multiple Comparisons with the Best (MCB) test.
    
    Parameters:
        data (array-like): Response values.
        groups (array-like): Group labels corresponding to each data point.
        seeds (array-like): Block labels (random effects) corresponding to each data point.
        alpha (float): Significance level (default 0.05).
        delta (float): Minimum percentage difference to consider (default 0.01 = 1%).
        delta_abs (float): Minimum absolute difference to consider (default 0.5).
    
    Returns:
        DataFrame: Comparison results with confidence intervals and significance.
    """
    # Convert to numpy arrays
    data = np.asarray(data)
    groups = np.asarray(groups)
    seeds = np.asarray(seeds)

    # Basic validation
    if len(data) != len(groups):
        raise ValueError("Data and group arrays must have the same length.")

    # Group statistics
    df = pd.DataFrame({'orig_value': data, 
                       'value': np.log(data),                        
                       'age': groups, 
                       'seed': seeds})
    df['age'] = df['age'].astype('category')

    # we assume that the age groups have different variances, especially true with lower ages, 
    # which can significantly skew the log-normal distribution if we assume homogeneous variance
    # use a glm with weights for each variance

    # Preliminary OLS to estimate residuals by group
    m0 = smf.ols("value ~ C(age) + C(seed)", data=df).fit()
    df["resid0"] = m0.resid

    # Residual variance per age
    group_var_resid = df.groupby("age", observed=True)["resid0"].var()

    # Stabilize: floor at median variance
    #floor = group_var_resid.median()
    #group_var_stable = np.maximum(group_var_resid, floor)
    group_var_stable = group_var_resid

    # WLS weights = 1 / variance
    df["weight"] = df["age"].map(lambda a: 1.0 / group_var_stable[a])


#    group_vars = df.groupby("age", observed=True)["value"].var()
    #df['weight'] = df['age'].map(lambda a: 1.0 / group_vars[a])
#    df['weight'] = df['age'].map(lambda a: group_vars[a])

    model = sm.WLS.from_formula(
         'value ~ C(age) + C(seed)',
         data=df,
         weights=df['weight']
    ).fit()

#     model = smf.ols(
# #        'value ~ C(age) + (1|seed)',
#          'value ~ C(age) + C(seed)',
#         data=df).fit()  # mixed effects model with seed as random effect (handles missing seeds)


    # LS-means via predict() on balanced grid
    A_levels = df['age'].cat.categories
    seed_levels = df['seed'].unique()
    mu_log = []
    for a in A_levels:
        grid = pd.DataFrame({
            'age': pd.Categorical([a] * len(seed_levels), categories=A_levels),
            'seed': list(seed_levels),
        })
        mu_log.append(model.predict(grid).mean())
    mu_log = pd.Series(mu_log, index=A_levels)

    best_level = mu_log.idxmin()

    # Extract group means and residual variance
    #orig_means = df.groupby('age', observed=True)['orig_value'].mean()
    #means = df.groupby('age', observed=True)["value"].mean()
    #A_levels = means.index
    #best_level = means.idxmin()

    # Residual variance and df
    #mse = model.mse_resid  # ols not weighted glm
    df_resid = model.df_resid

    rss_weighted = np.sum(model.resid ** 2 * model.model.weights)
    mse = rss_weighted / df_resid

    # Number of groups and replications
    k = len(A_levels)
    n_rep = df.groupby('age', observed=True)['seed'].nunique().mean()  # approx blocks per A

    # Critical value using Studentized range (Tukey)
    qcrit = qsturng(1 - alpha, k, df_resid)
    c_alpha = qcrit / np.sqrt(2)  # convert to t-like for pairwise diff

    # Compute SE for difference between two LSMeans
    se_diff = np.sqrt(2 * mse / n_rep)

    # Differences and upper bounds
    #diffs = means - means[best_level]
    diffs = mu_log - mu_log[best_level]

    UCB = diffs + c_alpha * se_diff
    UCB_pct = (np.exp(UCB) - 1)  # convert back to original scale    
    #UCB_abs = orig_means[best_level] * UCB_pct

    mu_raw_median = np.exp(mu_log)
    UCB_abs_median = mu_raw_median[best_level] * UCB_pct

    mu_raw_mean = mu_raw_median * np.exp(0.5 * mse)
    UCB_abs_mean = mu_raw_mean[best_level] * UCB_pct

    # Decide equivalence
    equiv = (UCB <= delta) | (UCB_abs_mean <= delta_abs) | (A_levels == best_level)  # percentage or absolute (seconds)
    candidates = [lvl for lvl in A_levels if equiv[lvl]]
    if best_level not in candidates:
        candidates.append(best_level)

    # Choose smallest A
    try:
        chosen = min(candidates, key=lambda x: float(x))
    except:
        chosen = candidates[0]

    return {
        "data": pd.DataFrame({'orig': mu_raw_mean, 'means': mu_log, 'diffs': diffs, 'UCB': UCB, 'UCB_abs': UCB_abs_mean, 'equivalent': equiv}),
        "best_level": best_level,
        "chosen_smallest_A": chosen,
        "critical_value": c_alpha,
        "SE_diff": se_diff,
        "delta": delta,
    }


COLUMNS = [
   "instance",
   "class",
   "machines",
   "jobs",
   "algorithm",
   "solver",
   "retention",
   "replication",
   "iterations",
   "lb",
   "ub",
   "gap",
   "obj",
   "redcost",
   "basis",
   "cols",
   "rmptime",
   "cgtime",
   "time",
   "lpiters",
   "lpiters_per_second",
   "lpiters_per_column",
   "has_integer",
   "timeout",
   "params",
   "last",
]

def main(argv=None):
    p = argparse.ArgumentParser(description="Compute aggregated GAP statistics.")
    p.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Input CSV file (with header).",
        default=Path(r"sweeps\lagrange-sweep-v3.csv"),
    )
    args = p.parse_args(argv)

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 2

    # Read experiment CSV
    try:
        df = pl.read_csv(
            args.input,
            infer_schema_length=100000,
            null_values=["", "NA", "NaN", "null", "inf"]
        )

    except Exception as e:
        print(f"Failed to read CSV: {e}", file=sys.stderr)
        return 3

    # Read optimal values from ../instances/gap/optimal_gap.csv
    # assumed columns: instance,rmp_opt,integer_opt
    try:
        optimal_df = pl.read_csv(
            Path(__file__).parent.parent.parent / "instances" / "gap" / "optimal_gap.csv",
            infer_schema_length=1000,
            null_values=["", "NA", "NaN", "null", "inf"]
        )

        df = df.join(optimal_df, on="instance", how="left")

    except Exception as e:
        print(f"Failed to read optimal values CSV: {e}", file=sys.stderr)
        return 4

    algorithm_order = ["Dantzig", "Wentges", "LT", "MT"]
    _alg_order_map = {name: i for i, name in enumerate(algorithm_order)}

    # rename algorithm values "LagrangeTemplate" -> "LT", "MipTemplate" -> "MT"
    df = df.with_columns([
        pl.col("algorithm")
          .str.replace("LagrangeTemplate", "LT")
          .str.replace("MipTemplate", "MT")
    ])

    # extract "age_limit" and "max_col_multiplier" json fields from params column
    df = df.with_columns([
        pl.col("params").str.extract(r".*age_limit:\s*(\d+)").cast(pl.Int64).alias("age_limit"),
        pl.col("params").str.extract(r".*random_seed:\s*(\d+)").cast(pl.Int64).alias("random_seed"),
        pl.col("params").str.extract(r".*max_col_multiplier:\s*(\d+)").cast(pl.Float64).alias("max_col_multiplier")
    ])

    df = df.with_columns(
        [(pl.col("jobs").cast(pl.Int64) // pl.col("machines").cast(pl.Int64)).alias("job_machine_ratio")]
    )

    instances = (df
                 .filter(pl.col("last") == 1)
                 .filter(pl.col("max_col_multiplier") == 1)
                 #.filter(pl.col("instance") == 'a05200')
                 #.filter(pl.col("timeout") == 0)
                 #.filter(pl.col('replication') < 5)
                 #.filter(pl.col('age_limit') > 2)
                 #.filter(pl.col('age_limit') < 10)
    )

    # remove all rows for an age_limit that has any timeouts
    # age_limits_with_timeouts = instances.filter(pl.col('timeout') > 0)['age_limit'].unique().implode()
    # instances = instances.filter(~pl.col('age_limit').is_in(age_limits_with_timeouts))

    # remove all instance that have timeouts for all age_limits
    instances_with_timeouts = instances.filter(pl.col('timeout') == 0)['instance'].unique().implode()
    instances = instances.filter(pl.col('instance').is_in(instances_with_timeouts))

    # group by instance and age_limit, compute mean time
    instances = instances.group_by(["instance", "age_limit", 'job_machine_ratio']).agg([(pl.col("time")).log().mean().exp().alias("mean_time")])
#    instances = instances.group_by(["instance", "age_limit", 'job_machine_ratio']).agg([pl.mean("time").alias("mean_time")])

    instances = instances.sort(['instance', 'age_limit'])
    
    # take rolling mean of time over age_limit to smooth out noise
    instances = instances.with_columns([
        pl.col("mean_time").rolling_mean(window_size=5, weights=[1,2,4,2,1], min_samples=3, center=True).over('instance').alias("time_smoothed")
    ])
  
    instances.write_csv("sweep_raw.csv")
    
    # find the row with the minimum mean_time for each instance
    best_times = instances.sort(["instance", "time_smoothed"]).group_by(["instance", 'job_machine_ratio']).first()

    # find the smallest age_limit that is within 5% of the best mean_time for each instance
    instances = instances.join(best_times, on="instance", how="left")
    instances = instances.with_columns(
        (pl.col("time_smoothed") <= 1.01 * pl.col("time_smoothed_right")).alias("within_5_percent"),
        (pl.col("time_smoothed") - pl.col("time_smoothed_right")).alias("abs_diff")
    )

    instances = instances.filter(pl.col("within_5_percent") | (pl.col("abs_diff") <= 1)) # also within 0.5 seconds
    instances = instances.group_by(["instance", 'job_machine_ratio']).agg(
        pl.col("age_limit").min().alias("best_age")
    )

    # sort by job_machine_ratio then best_age
    instances = instances.sort(['job_machine_ratio', 'best_age'])
    print(instances)

    # want to find the tightest quadratic curve that lies above all points (job_machine_ratio, best_age)
    max_instances = instances.group_by(pl.col('job_machine_ratio')).agg([
        pl.col('best_age').max().alias('best_age')
    ])

    import highspy
    h = highspy.Highs()
    [a,b,c] = h.addVariables(3)
    max_pt = h.addVariable()

    h.addConstrs(
        y <= a * x * x + b * x + c for (x,y) in zip(max_instances['job_machine_ratio'], max_instances['best_age'])
    )
    h.addConstr(c >= 1)  # age limit at least 1
    h.addConstrs(
        max_pt >= a * x * x + b * x + c - y for (x,y) in zip(max_instances['job_machine_ratio'], max_instances['best_age'])
    )
    h.minimize(max_pt)
    print(h.vals([a,b,c,max_pt]))
    
    # gurobi version, but minimize quadratic distance
    from gurobipy import Model, GRB
    m = Model("age_limit_fit")
    a = m.addVar(name="a")
    b = m.addVar(name="b")
    c = m.addVar(name="c")
    max_dev = m.addVar(name="max_dev")
    m.setObjective(max_dev, GRB.MINIMIZE)
    for x, y in zip(max_instances['job_machine_ratio'], max_instances['best_age']):
        expr = a * x * x + b * x + c
        m.addConstr(expr >= y, name=f"cover_{x}_{y}")
        m.addConstr(c >= 1, name="min_c")

    m.addConstr(max_dev == sum((a * x * x + b * x + c - y) * (a * x * x + b * x + c - y) for (x,y) in zip(max_instances['job_machine_ratio'], max_instances['best_age'])))
    m.optimize()

    print(a.X, b.X, c.X, max_dev.X)


    # add fitted age_limit to instances
#    a_val, b_val, c_val = h.vals([a,b,c])
    a_val, b_val, c_val = a.X, b.X, c.X
    instances = instances.with_columns(
        (a_val * pl.col('job_machine_ratio') ** 2 + b_val * pl.col('job_machine_ratio') + c_val).alias('fitted_age_limit')
    )

    # write to CSV
    instances.write_csv("sweep_best_age_limits.csv")
    return 0

    # get unique instances
    test = sorted(instances.unique(subset=["instance"])['instance'])

    # testing a single instance
    #
    inst = 'a05100'
    print(f"Processing instance: {inst}")
    tmp = instances.filter(pl.col("instance") == inst)

    # # remove all rows for an age_limit that has any timeouts
    # age_limits_with_timeouts = tmp.filter(pl.col('timeout') > 0)['age_limit'].unique().implode()
    # tmp = tmp.filter(~pl.col('age_limit').is_in(age_limits_with_timeouts))

    #penalty = tmp['time'].max()

    # print(instances)
    tmp.write_csv(f"sweep_statistics-lagrange-{inst}.csv")

    # Example dataset 
    #result_df = hsu_mcb(tmp["time"] + penalty * tmp["timeout"], tmp["age_limit"], tmp["random_seed"], alpha=0.05, delta=0.6, delta_abs=0.5)
    result_df = hsu_mcb(tmp["time"], tmp["age_limit"], tmp["random_seed"], alpha=0.1, delta=0.4, delta_abs=0.5)

    with pd.option_context('display.max_rows', None):
        for key, value in result_df.items():
            print(f"{key}: {value}")

    print("Hsu's MCB chosen A (smallest within 40% of best):", result_df['chosen_smallest_A'])
    return 0


    results_instance = []
    results_jobs_machine_ratio = []
    results_chosen = []

    for inst in test:
        print(f"Processing instance: {inst}")
        tmp = instances.filter(pl.col("instance") == inst)
        # penalty = tmp['time'].max()

        # remove all rows for an age_limit that has any timeouts
        age_limits_with_timeouts = tmp.filter(pl.col('timeout') > 0)['age_limit'].unique().implode()
        tmp = tmp.filter(~pl.col('age_limit').is_in(age_limits_with_timeouts))

        if tmp.is_empty():
            print(f"Error: {inst} has timeouts for all records")
            continue

        # Example dataset
        #result_df = hsu_mcb(tmp["time"] + penalty * tmp['timeout'], tmp["age_limit"], tmp["random_seed"], alpha=0.05, delta=0.6, delta_abs=0.5)
        result_df = hsu_mcb(tmp["time"], tmp["age_limit"], tmp["random_seed"], alpha=0.05, delta=0.4, delta_abs=0.5)

        if tmp.filter(pl.col('age_limit') == result_df['chosen_smallest_A'])['timeout'].sum() > 0:
            print(f"Error: chosen A={result_df['chosen_smallest_A']} has timeouts for instance {inst}")
        else:
            results_instance.append(inst)
            results_jobs_machine_ratio.append(
                float(tmp["job_machine_ratio"].unique()[0]))
            results_chosen.append(result_df['chosen_smallest_A'])

    # save results to csv
    results_df = pd.DataFrame({
        "algorithm": instances["algorithm"].unique()[0],
        "instance": results_instance,
        "job_machine_ratio": results_jobs_machine_ratio,
        "best_age": results_chosen
    })
    # sort by job_machine_ratio
    results_df = results_df.sort_values(by="job_machine_ratio")
    results_df.to_csv(f"sweep_statistics-{instances["algorithm"].unique()[0]}.csv", index=False)

    print(results_df)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())