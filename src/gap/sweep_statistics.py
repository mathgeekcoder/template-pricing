import polars as pl
import numpy as np
from highspy import Highs, HessianFormat
from utils import load_results_csv, extract_parameters

def load_and_smooth(sweep_file) -> pl.DataFrame:
    instances = (
        extract_parameters(load_results_csv("parameter-sweeps", sweep_file))
            .filter(pl.col("last") == 1)
            .filter(pl.col("max_col_multiplier") == 1)
    )

    # remove all instances that have timeouts for all age_limits
    instances_with_timeouts = instances.filter(pl.col('timeout') == 0)['instance'].unique().implode()
    instances = instances.filter(pl.col('instance').is_in(instances_with_timeouts))

    # group by instance and age_limit, compute geometric mean time
    instances = instances.group_by([
        "instance", "age_limit", 'job_machine_ratio'
    ]).agg([pl.col("time").log().mean().exp().alias("mean_time")])

    instances = instances.sort(['instance', 'age_limit'])

    # take rolling mean of time over age_limit to smooth out noise
    return instances.with_columns([
        pl.col("mean_time")
          .rolling_mean(window_size=5, weights=[1, 2, 4, 2, 1], min_samples=3, center=True)
          .over(['instance']).alias("time_smoothed")
    ])


def find_best_age_limits(instances: pl.DataFrame) -> pl.DataFrame:
    # find the row with the minimum mean_time for each instance
    best_times = (instances.sort(["instance", "time_smoothed"])
                           .group_by(["instance", 'job_machine_ratio'])
                           .first())

    # find the smallest age_limit that is within 1% of the best mean_time for each instance
    instances = instances.join(best_times, on="instance", how="left")
    instances = instances.with_columns(
        (pl.col("time_smoothed") <= 1.01 * pl.col("time_smoothed_right")).alias("within_1_percent"),
        (pl.col("time_smoothed") - pl.col("time_smoothed_right")).alias("abs_diff"))

    instances = instances.filter(
        pl.col("within_1_percent") | (pl.col("abs_diff") <= 1))  # also within 1 second

    instances = instances.group_by(["instance", 'job_machine_ratio']).agg(
        pl.col("age_limit").min().alias("best_age"))

    # sort by job_machine_ratio then best_age
    return instances.sort(['job_machine_ratio', 'best_age'])


# Find the age limit policy as a function of job_machine_ratio
#
# want to find the tightest quadratic curve that lies above all points (job_machine_ratio, best_age)
def find_age_limit_policy(instances: pl.DataFrame, method: str):
    max_instances = (
        instances.group_by(pl.col('job_machine_ratio'))
                 .agg([
                     pl.col('best_age')
                       .max()
                       .alias('best_age')
                 ])
        )

    ratio = list(max_instances['job_machine_ratio'])
    best_age = list(max_instances['best_age'])

    h = Highs()
    h.silent()

    [a, b] = h.addVariables(2)
    delta = h.addVariables(len(ratio))

    h.addConstrs(delta[i] + y == a *x*x + b *x + 1 for i, (x, y) in enumerate(zip(ratio, best_age)))

    # minimize quadratic, i.e., sum(delta[i] * delta[i])
    h.passHessian(len(ratio)+2, len(ratio), HessianFormat.kTriangular,
                  [0,0] + list(range(len(ratio))), range(2, len(ratio)+2), [2.0] * len(ratio))

    h.minimize()
    print(f"{method}: {h.val(a):.6f} x^2 + {h.val(b):.4f} x + 1\n")



def process_sweep_csv(method, sweep_file) -> pl.DataFrame:
    instances = load_and_smooth(sweep_file)
    instances = find_best_age_limits(instances)
    print(instances)

    find_age_limit_policy(instances, method)

    # write to CSV
    instances.write_csv(f"best-age-limits-{method}.csv")


if __name__ == "__main__":
    pl.Config.set_tbl_cell_alignment("RIGHT")
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    process_sweep_csv("dantzig", "age-sweep-dantzig.csv")
    process_sweep_csv("pessoa", "age-sweep-pessoa.csv")
    process_sweep_csv("lt", "age-sweep-lt.csv")


    # output single instance as an example
    a05200 = (
        extract_parameters(load_results_csv("parameter-sweeps", "age-sweep-dantzig.csv"))
            .filter(pl.col("last") == 1)
            .filter(pl.col("max_col_multiplier") == 1)
            .filter(pl.col('instance') == 'a05200')
            .select(["replication", "age_limit", "time"])
    )

    # join on smoothed times for a05200
    a05200_smoothed = (
        load_and_smooth("age-sweep-dantzig.csv")
            .filter(pl.col('instance') == 'a05200')
            .select(['age_limit', 'time_smoothed'])
    )

    # this will duplicate for each replication, but that's fine for example purposes
    a05200 = a05200.join(other=a05200_smoothed, on="age_limit", how="left")

    a05200.write_csv(f"a05200-age-limits-dantzig.csv")
    print(a05200)
