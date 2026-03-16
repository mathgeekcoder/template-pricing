#!/usr/bin/env python
from __future__ import annotations
import polars as pl
from utils import load_results_csv, sort_by_algorithm

def farkas(df):
    # -1 and 6 correspond initialization (6 being integer feasible LP)
    # -4 is timeout, which we also want to include in the analysis
    farkas = df.filter(pl.col("last").is_in([-1, 6, -4]))

    farkas = (
        farkas
          .group_by(["farkas"])
          .agg([
              (pl.col("rmptime") * 1000).mean().round(1).alias("RMP (ms)"),
              (pl.col("cgtime") * 1000).mean().round(1).alias("Pricing (ms)"),
              pl.col("iterations").mean().round(1).alias("#its"),
              (
                pl.when(pl.col("obj").is_not_null())
                  .then((pl.col("obj") - pl.col("rmp_opt")) / pl.col("rmp_opt"))
                  .otherwise(None)
                  .mean() * 100
              ).round(1).alias("%gap"),
              (pl.col("has_integer") * 100).mean().round(1).alias("%integer"),
              (pl.col("timeout") * 100).mean().round(1).alias(f"%timeout"),
              pl.len().alias("count"),

          ])
          .sort(["farkas"])
    )

    print(farkas)
    print()


def pivot_times(df, group_by = ["class", "jobs", "machines"]):
    df = df.filter(pl.col("last") == 1)

    df = (
        df.group_by(["algorithm"] + group_by).agg([
            pl.col("rmptime").mean().round(1).alias("RMP"),
            pl.col("cgtime").mean().round(1).alias("CG"),
            pl.col("time").log().mean().exp().alias("time"),
            pl.len().alias("count"),
        ])
        .sort(group_by)
    )

    # compute best time per instance, and produce multipliers for each algorithm relative to the best time
    df = df.with_columns([
        pl.col("time").min().over(group_by).alias("best_time"),
    ]).with_columns([(pl.col("time") /
                      pl.col("best_time")).round(0).alias("time_multiplier")])


    pivot = df.pivot(
        values=["RMP", "CG", "time_multiplier", "count"],
        index=group_by,
        on="algorithm"
    )

    return pivot

if __name__ == "__main__":
    pl.Config.set_tbl_cell_alignment("RIGHT")
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_cols(-1)

    # COLUMNS = ["instance", "class", "machines", "jobs", "algorithm", "solver", "retention", "replication", "iterations", "lb", "ub", "gap", "obj", "redcost", "basis", "cols", "rmptime", "cgtime", "time", "lpiters", "lpiters_per_second", "lpiters_per_column", "has_integer", "timeout", "last"]
    df = load_results_csv("yagiura", "yagiura-results.csv")

    # Read optimal values
    optimal_df = load_results_csv("yagiura", "yagiura-optimal.csv")
    df = df.join(optimal_df, on="instance", how="left")

    # farkas initialization
    print("Farkas set cover:")
    init_setcover = load_results_csv("yagiura", "yagiura-init-setcover.csv")
    init_setcover = init_setcover.join(optimal_df, on="instance", how="left")
    farkas(init_setcover)

    print("Farkas set partition:")
    init_partition = load_results_csv("yagiura", "yagiura-init-setpartition.csv")
    init_partition = init_partition.join(optimal_df, on="instance", how="left")
    farkas(init_partition)


    # solved vs time for figure
    print("Solved vs time:")

    # capture cumulative count of solved vs time for each algorithm,
    # where time is quantized in minutes and cumulative count is number of instances solved within that time
    solved = df.filter(pl.col("last") == 1)

    solved = solved.with_columns([
        pl.when(pl.col("time") > 21600)
          .then(21600*2)
          .otherwise(pl.col("time"))
          .alias("time_cap")
    ])

    # quantize time to 0.5 minutes
    solved = solved.with_columns([
        ((pl.col("time_cap") // 30) / 2.0).cast(pl.Float64).alias("time_minutes")
    ])

    # create a complete range of time_minutes from 0 to 360 in 0.5 increments
    all_time_minutes = pl.DataFrame({
        "time_minutes": [i / 2.0 for i in range(0, 721)]  # 0, 0.5, 1, ..., 360
    })

    # group by algorithm and time_minutes, count instances, then compute cumulative sum
    solved = (solved.group_by(["algorithm", "time_minutes"]).agg([
        pl.len().alias("count")
    ]).sort(["algorithm", "time_minutes"]).with_columns(
        pl.col("count").cum_sum().over("algorithm").alias("cumsum")).pivot(
            values="cumsum", index="time_minutes",
            on="algorithm").sort("time_minutes"))

    # join with all_time_minutes to ensure no gaps, then forward fill
    solved = (all_time_minutes
        .join(solved, on="time_minutes", how="left")
        .sort("time_minutes")
        .fill_null(strategy="forward")
        .fill_null(0)  # fill any remaining nulls at the start with 0
        .filter(pl.col("time_minutes") <= 360))

    # convert cumsum to percentage, i.e., divide by 285 instances
    solved = solved.with_columns([
        (pl.col("D") / 285 * 100).round(1).alias("D"),
        (pl.col("P") / 285 * 100).round(1).alias("P"),
        (pl.col("LT") / 285 * 100).round(1).alias("LT"),
        (pl.col("MT") / 285 * 100).round(1).alias("MT"),
    ]).select(["time_minutes", "D", "P", "LT", "MT"])

    solved.write_csv("yagiura-solved-vs-time.csv")
    print(solved)
    print()

    print("Aggregate statistics:")
    stats = df.filter(pl.col("last") == 1)

    stats = sort_by_algorithm(
        stats
          .group_by(["algorithm"])
          .agg([
              (pl.col("timeout").mean() * 100).round(1).alias("%timeout"),
              pl.col("iterations").mean().round(0).alias("#its"),
              pl.col("rmptime").mean().round(1).alias("RMP (s)"),
              pl.col("cgtime").mean().round(1).alias("Pricing (s)"),

              pl.col("lpiters_per_column").mean().round(1).alias("pivots/col"),
              (pl.col("lpiters").sum() / pl.col("rmptime").sum()).round(1).alias("pivots/s"),
              (pl.col("has_integer").mean() * 100).round(1).alias("%integral"),
              pl.col("gap").mean().round(1).alias("%gap"),
              pl.len().alias("count"),
          ])
    )

    stats.write_csv("yagiura-stats.csv")
    print(stats)
    print()

    # job/machine times
    print("Job/Machine:")
    job_machine = pivot_times(df.filter(pl.col("last") == 1), ['jobs', 'machines'])

    job_machine.write_csv("yagiura-job-machine.csv")
    print(job_machine)
    print()


    # class/job/machine times
    print("All:")
    class_job_machine = pivot_times(df, ['class', 'jobs', 'machines'])

    class_job_machine.write_csv("yagiura-class-job-machine.csv")
    print(class_job_machine)
