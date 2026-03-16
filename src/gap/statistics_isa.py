#!/usr/bin/env python
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
              pl.col("rmptime").mean().round(1).alias("RMP (s)"),
              pl.col("cgtime").mean().round(1).alias("Pricing (s)"),
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

if __name__ == "__main__":
    pl.Config.set_tbl_cell_alignment("RIGHT")
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_cols(-1)

    df = load_results_csv("isa", "isa-results.csv")

    # Read optimal values
    optimal_df = load_results_csv("isa", "isa-optimal.csv")
    df = df.join(optimal_df, on="instance", how="left")

    # Consider not-solved if:
    # P, LT: timeout == 1
    #    LR: timeout == 1 or rmp_opt != LB (since we treat non-optimal LR solutions as timeouts)
    df = df.with_columns([
        ((pl.col("timeout") == 1) | ((pl.col("algorithm") == "LR") & (pl.col("rmp_opt") != (pl.col("lb") - 1e-6).ceil()))).alias("notsolved")
    ])


    # farkas initialization
    print("Farkas set cover:")
    init_setcover = load_results_csv("isa", "isa-init-setcover.csv")
    init_setcover = init_setcover.join(optimal_df, on="instance", how="left")
    farkas(init_setcover)


    # solved vs time for figure
    print("Solved vs time:")

    # capture cumulative count of solved vs time for each algorithm,
    # where time is quantized in minutes and cumulative count is number of instances solved within that time
    solved = (df
        .filter(pl.col("last") == 1)
        .filter(pl.col("notsolved") == 0)
    )

    # quantize time to 0.5 minutes
    solved = solved.with_columns([
        ((pl.col("time") // 30) / 2.0).cast(pl.Float64).alias("time_minutes")
    ])

    # create a complete range of time_minutes from 0 to 60 in 0.5 increments
    all_time_minutes = pl.DataFrame({
        "time_minutes": [i / 2.0 for i in range(0, 121)]  # 0, 0.5, 1, ..., 60
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
        .filter(pl.col("time_minutes") <= 60))

    # convert cumsum to percentage, i.e., divide by 1735 instances
    solved = solved.with_columns([
        (pl.col("P") / 1735 * 100).round(1).alias("P"),
        (pl.col("LT") / 1735 * 100).round(1).alias("LT"),
        (pl.col("LR") / 1735 * 100).round(1).alias("LR"),
    ]).select(["time_minutes", "P", "LT", "LR"])

    solved.write_csv("isa-solved-vs-time.csv")
    print(solved)
    print()


    print("Aggregate statistics:")
    stats = df.filter(pl.col("last") == 1)

    stats = sort_by_algorithm(
        stats
          .group_by(["algorithm"])
          .agg([
              (100 - pl.col("notsolved").mean() * 100).round(1).alias("% solved"),
              pl.col("iterations").mean().round(0).alias("#its"),
              pl.col("rmptime").mean().round(1).alias("RMP (s)"),
              pl.col("cgtime").mean().round(1).alias("Pricing (s)"),

              pl.col("lpiters_per_column").mean().round(1).alias("pivots/col"),
              (pl.col("lpiters").sum() / pl.col("rmptime").sum()).round(1).alias("pivots/s"),
              (pl.col("has_integer").mean() * 100).round(1).alias("%integral"),
              pl.col("gap").mean().round(1).alias("%gap"),
              pl.len().alias("count")
          ])
    )

    stats.write_csv("isa-stats.csv")
    print(stats)
    print()


    print("Filtered Aggregate statistics:")
    stats = df.filter(pl.col("last") == 1)

    ## only consider instances solved by all algorithms, to compare solution quality and runtime on the same set of instances
    solved_instances = stats.group_by('instance').agg(pl.col("notsolved").sum().alias("notsolved")).filter(pl.col("notsolved") == 0).select("instance").unique()['instance'].to_list()
    stats = stats.filter(pl.col("instance").is_in(solved_instances))

    stats = sort_by_algorithm(
        stats
          .group_by(["algorithm"])
          .agg([
              (100 - pl.col("notsolved").mean() * 100).round(1).alias("% solved"),
              pl.col("iterations").mean().round(0).alias("#its"),
              pl.col("rmptime").mean().round(1).alias("RMP (s)"),
              pl.col("cgtime").mean().round(1).alias("Pricing (s)"),

              pl.col("lpiters_per_column").mean().round(1).alias("pivots/col"),
              (pl.col("lpiters").sum() / pl.col("rmptime").sum()).round(1).alias("pivots/s"),
              (pl.col("has_integer").mean() * 100).round(1).alias("%integral"),
              pl.col("gap").mean().round(1).alias("%gap"),
              pl.len().alias("count")
          ])
    )

    print(stats)
    print()


    # class/job statistics
    print("Class/Job:")
    class_job = df.filter(pl.col("last") == 1)

    class_job = (
        class_job
         .group_by(["algorithm", "class", "jobs"]).agg([
            pl.col("rmptime").mean().round(1).alias("RMP"),
            pl.col("cgtime").mean().round(1).alias("CG"),
            (pl.col("has_integer").mean() * 100).round(0).alias("integral"),
            (pl.col("notsolved").mean() * 100).round(0).alias("timeout"),
            pl.len().alias("count"),
        ])
        .sort("class", "jobs")
    )

    # pivot by class and jobs, with columns for each algorithm, values are avg hasopt, runtime_cap, cgtime, has_integer
    class_job_pivot = class_job.pivot(
        values=["RMP", "CG", "integral", "timeout", "count"],
        index=["class", "jobs"],
        on="algorithm"
    )

    class_job_pivot.write_csv("isa-class-job.csv")
    print(class_job_pivot)
