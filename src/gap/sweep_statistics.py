import argparse
import sys
from pathlib import Path
import polars as pl

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

def load_sweep_csv(input_path: Path) -> pl.DataFrame:
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        exit(2)

    # Read experiment CSV
    try:
        df = pl.read_csv(input_path,
                         infer_schema_length=100000,
                         null_values=["", "NA", "NaN", "null", "inf"])

    except Exception as e:
        print(f"Failed to read CSV: {e}", file=sys.stderr)
        exit(3)

    # Read optimal values from ../instances/gap/optimal_gap.csv
    # assumed columns: instance,rmp_opt,integer_opt
    try:
        optimal_df = pl.read_csv(Path(__file__).parent.parent.parent /
                                 "instances" / "gap" / "optimal_gap.csv",
                                 infer_schema_length=1000,
                                 null_values=["", "NA", "NaN", "null", "inf"])

        df = df.join(optimal_df, on="instance", how="left")

    except Exception as e:
        print(f"Failed to read optimal values CSV: {e}", file=sys.stderr)
        exit(4)

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

    df = df.with_columns([
        (pl.col("jobs").cast(pl.Int64) // pl.col("machines").cast(pl.Int64)).alias("job_machine_ratio")
    ])

    return (
        df.filter(pl.col("last") == 1)
          .filter(pl.col("max_col_multiplier") == 1)
    )


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Compute aggregated GAP statistics.")

    p.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Input CSV file (with header).",
        default=Path(r"sweeps\lagrange-sweep-v3.csv"),
    )

    args = p.parse_args(argv)
    instances = load_sweep_csv(args.input)

    #instances = instances.filter(pl.col('instance') == 'a05100')

    # remove all instances that have timeouts for all age_limits
    instances_with_timeouts = instances.filter( pl.col('timeout') == 0)['instance'].unique().implode()
    instances = instances.filter(pl.col('instance').is_in(instances_with_timeouts))

    # group by instance and age_limit, compute mean time
    instances = instances.group_by([
        "instance", "age_limit", 'job_machine_ratio'
    ]).agg([pl.col("time").log().mean().exp().alias("mean_time")])

    instances = instances.sort(['instance', 'age_limit'])
    instances.write_csv("sweep_raw.csv")

    # take rolling mean of time over age_limit to smooth out noise
    instances = instances.with_columns([
        pl.col("mean_time")
          .rolling_mean(window_size=5, weights=[1, 2, 4, 2, 1], min_samples=3, center=True)
          .over(['instance']).alias("time_smoothed")
    ])

    instances.write_csv("sweep_raw_1.csv")

    # find the row with the minimum mean_time for each instance
    best_times = (instances.sort(["instance", "time_smoothed"])
                           .group_by(["instance", 'job_machine_ratio'])
                           .first())

    # find the smallest age_limit that is within 1% of the best mean_time for each instance
    instances = instances.join(best_times, on="instance", how="left")
    instances = instances.with_columns(
        (pl.col("time_smoothed") <= 1.01 * pl.col("time_smoothed_right")).alias("within_1_percent"),
        (pl.col("time_smoothed") - pl.col("time_smoothed_right")).alias("abs_diff"))

    instances.write_csv("sweep_raw_2.csv")

    instances = instances.filter(
        pl.col("within_1_percent") | (pl.col("abs_diff") <= 1))  # also within 0.5 seconds

    instances = instances.group_by(["instance", 'job_machine_ratio']).agg(
        pl.col("age_limit").min().alias("best_age"))

    # sort by job_machine_ratio then best_age
    instances = instances.sort(['job_machine_ratio', 'best_age'])
    print(instances)
    instances.write_csv("sweep_raw_3.csv")

    # Find the age limit policy as a function of job_machine_ratio
    #
    # want to find the tightest quadratic curve that lies above all points (job_machine_ratio, best_age)
    max_instances = instances.group_by(pl.col('job_machine_ratio')).agg(
        [pl.col('best_age').max().alias('best_age')])

    import highspy
    h = highspy.Highs()
    [a, b, c] = h.addVariables(3)
    max_pt = h.addVariable()

    h.addConstrs(y <= a * x * x + b * x + c for (x, y) in zip(
        max_instances['job_machine_ratio'], max_instances['best_age']))
    h.addConstr(c >= 1)  # age limit at least 1
    h.addConstrs(max_pt >= a * x * x + b * x + c - y for (x, y) in zip(
        max_instances['job_machine_ratio'], max_instances['best_age']))
    h.minimize(max_pt)
    print(h.vals([a, b, c, max_pt]))

    # # gurobi version, but minimize quadratic distance
    # from gurobipy import Model, GRB
    # m = Model("age_limit_fit")
    # a = m.addVar(name="a")
    # b = m.addVar(name="b")
    # c = m.addVar(name="c")
    # max_dev = m.addVar(name="max_dev")
    # m.setObjective(max_dev, GRB.MINIMIZE)
    # for x, y in zip(max_instances['job_machine_ratio'],
    #                 max_instances['best_age']):
    #     expr = a * x * x + b * x + c
    #     m.addConstr(expr >= y, name=f"cover_{x}_{y}")
    #     m.addConstr(c >= 1, name="min_c")

    # m.addConstr(max_dev == sum(
    #     (a * x * x + b * x + c - y) * (a * x * x + b * x + c - y)
    #     for (x, y) in zip(max_instances['job_machine_ratio'],
    #                       max_instances['best_age'])))
    # m.optimize()

    # print(a.X, b.X, c.X, max_dev.X)

    # # add fitted age_limit to instances
    # #    a_val, b_val, c_val = h.vals([a,b,c])
    # a_val, b_val, c_val = a.X, b.X, c.X
    # instances = instances.with_columns(
    #     (a_val * pl.col('job_machine_ratio')**2 +
    #      b_val * pl.col('job_machine_ratio') +
    #      c_val).alias('fitted_age_limit'))

    # write to CSV
    instances.write_csv("sweep_best_age_limits.csv")
    return 0

    # testing a single instance
    #
    inst = 'a05100'
    print(f"Processing instance: {inst}")
    tmp = instances.filter(pl.col("instance") == inst)

    # print(instances)
    tmp.write_csv(f"sweep_statistics-lagrange-{inst}.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())