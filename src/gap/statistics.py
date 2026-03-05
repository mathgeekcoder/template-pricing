#!/usr/bin/env python
from __future__ import annotations

import math
import sys
from pathlib import Path
import polars as pl

def gap_expr(column: str) -> pl.Expr:
    """
    Build a Polars expression computing average relative gap to optimal rmp solution.
    """
    return (
        pl.when(pl.col(column).is_not_null())
          .then((pl.col(column) - pl.col("rmp_opt")) / pl.col("rmp_opt"))
          .otherwise(None)
          .mean()
          .alias("avg_rel_gap")
    )

def load_csv(file):
    try:
        df = pl.read_csv(
            Path(__file__).parent.parent.parent / "results" / "yagiura" / file,
            infer_schema_length=1000,
            null_values=["", "NA", "NaN", "null", "inf"]
        )

    except Exception as e:
        print(f"Failed to read CSV: {e}", file=sys.stderr)
        sys.exit(3)

    return df

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
              (gap_expr("obj") * 100).round(1).alias("%gap"),
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

    # COLUMNS = ["instance", "class", "machines", "jobs", "algorithm", "solver", "retention", "replication", "iterations", "lb", "ub", "gap", "obj", "redcost", "basis", "cols", "rmptime", "cgtime", "time", "lpiters", "lpiters_per_second", "lpiters_per_column", "has_integer", "timeout", "last"]
    df = load_csv("yagiura-results.csv")

    # Read optimal values
    optimal_df = load_csv("yagiura-optimal.csv")
    df = df.join(optimal_df, on="instance", how="left")

    algorithm_order = ["D", "P", "LT", "MT"]
    _alg_order_map = {name: i for i, name in enumerate(algorithm_order)}

    # rename algorithm values "LagrangeTemplate" -> "LT", "MipTemplate" -> "MT"
    df = df.with_columns([
        pl.col("algorithm")
          .str.replace("Dantzig", "D")
          .str.replace("Wentges", "P")
          .str.replace("LagrangeTemplate", "LT")
          .str.replace("MipTemplate", "MT")
    ])


    # farkas initialization
    print("Farkas set cover:")
    init_setcover = load_csv("yagiura-init-setcover.csv")
    init_setcover = init_setcover.join(optimal_df, on="instance", how="left")
    farkas(init_setcover)

    print("Farkas set partition:")
    init_partition = load_csv("yagiura-init-setpartition.csv")
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
        ((pl.col("time_cap") // 30) / 2.0 + 1).cast(pl.Float64).alias("time_minutes")
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

    # convert cumsum to percentage, i.e., divide by 1735 instances
    solved = solved.with_columns([
        (pl.col("D") / 285 * 100).round(1).alias("D"),
        (pl.col("P") / 285 * 100).round(1).alias("P"),
        (pl.col("LT") / 285 * 100).round(1).alias("LT"),
        (pl.col("MT") / 285 * 100).round(1).alias("MT"),
    ]).select(["time_minutes", "D", "P", "LT", "MT"])

    #solved.write_csv("agg_solved.csv")
    print(solved)
    print()

    print("Aggregate statistics:")
    stats = df.filter(pl.col("last").is_in([1])).filter(pl.col("rmp_opt").is_not_null())

    # replace "#NAME?" or "nan" in "gap" column with null, then cast to float
    stats = stats.with_columns([
        pl.col("gap").cast(pl.String).str.replace_all(r"#NAME\?|nan|^$", "").alias("gap")
    ]).with_columns([
        pl.when(pl.col("gap") == "").then(None).otherwise(
            pl.col("gap").cast(pl.Float64, strict=False)).alias("gap")
    ])

    stats = (
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
          .with_columns(
              pl.col("algorithm").replace_strict(_alg_order_map, default=999).alias("_alg_order")
          )
          .sort(["_alg_order"])
          .drop(["_alg_order"])
    )

    #stats.write_csv("agg_stats.csv")
    print(stats)
    print()

    print("Job/Machine:")

    # job/machine status
    job_machine = df.filter(pl.col("last") == 1)

    job_machine = (
        job_machine
         .group_by(["algorithm", "jobs", "machines"]).agg([
            pl.col("rmptime").mean().round(1).alias("RMP"),
            pl.col("cgtime").mean().round(1).alias("CG"),
            pl.len().alias("count"),
        ])
         .with_columns(
             pl.col("algorithm").replace_strict(_alg_order_map, default=999).alias("_alg_order")
         )
         .sort("_alg_order", "jobs", "machines")
         .drop("_alg_order")
    )

    # pivot by class and jobs, with columns for each algorithm, values are avg hasopt, runtime_cap, cgtime, has_integer
    job_machine_pivot = job_machine.pivot(
        values=["RMP", "CG", "count"],
        index=["jobs", "machines"],
        on="algorithm"
    )

#    job_machine_pivot.write_csv("agg_job_machine.csv")
    print(job_machine_pivot)
    print()


    print("All:")

    # class/job/machine status
    class_job_machine = df.filter(pl.col("last") == 1)

    class_job_machine = (
        class_job_machine
         .group_by(["algorithm", "class", "jobs", "machines"]).agg([
            pl.col("rmptime").mean().round(1).alias("RMP"),
            pl.col("cgtime").mean().round(1).alias("CG"),
            pl.len().alias("count"),
        ])
         .with_columns(
             pl.col("algorithm").replace_strict(_alg_order_map, default=999).alias("_alg_order")
         )
         .sort("_alg_order", "class", "jobs", "machines")
         .drop("_alg_order")
    )

    # pivot by class and jobs, with columns for each algorithm, values are avg hasopt, runtime_cap, cgtime, has_integer
    class_job_machine_pivot = class_job_machine.pivot(
        values=["RMP", "CG", "count"],
        index=["class", "jobs", "machines"],
        on="algorithm"
    )

    #class_job_machine_pivot.write_csv("agg_class_job_machine.csv")
    print(class_job_machine_pivot)
