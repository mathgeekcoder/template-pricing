#!/usr/bin/env python
from __future__ import annotations

import argparse
from ast import alias
import math
from socket import timeout
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
   "farkas",
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
   "last",
]

def geo_expr(column: str) -> pl.Expr:
    """
    Build a Polars expression computing geometric mean of positive values in 'column'.
    Excludes non-positive entries.
    """
    col = pl.col(column)
    return col.filter(col > 0).log().mean().exp().alias(f"geo_{column}")

def avg_expr(column: str) -> pl.Expr:
    """
    Build a Polars expression computing mean of values in 'column'.
    """
    return pl.col(column).mean().alias(f"avg_{column}")

def sum_expr(column: str) -> pl.Expr:
    return pl.col(column).sum().alias(f"sum_{column}")

def std_expr(column: str) -> pl.Expr:
    """
    Build a Polars expression computing standard deviation of values in 'column'.
    """
    return pl.col(column).std().alias(f"std_{column}")

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

def boxplot_expr(column: str) -> pl.Expr:
    """
    Build a Polars expression computing boxplot statistics of values in 'column'.
    """
    return [
        pl.col(column).min().alias(f"min_{column}"),
        pl.col(column).quantile(0.25).alias(f"q1_{column}"),
        pl.col(column).median().alias(f"median_{column}"),
        pl.col(column).quantile(0.75).alias(f"q3_{column}"),
        pl.col(column).max().alias(f"max_{column}")
    ]

def orderby(column: str) -> pl.Expr:
    return pl.col(column).mean().alias(f"_order")

def main(argv=None):
    p = argparse.ArgumentParser(description="Compute aggregated GAP statistics.")
    p.add_argument("input", type=Path, help="Input CSV file (with header).")
    args = p.parse_args(argv)

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 2

    # Read experiment CSV
    try:
        df = pl.read_csv(
            args.input,
            infer_schema_length=1000,
            null_values=["", "NA", "NaN", "null", "inf"]
        )

    except Exception as e:
        print(f"Failed to read CSV: {e}", file=sys.stderr)
        return 3

    # Read optimal values from ../results/isa-optimal.csv
    # assumed columns: instance,rmp_opt
    try:
        optimal_df = pl.read_csv(
            Path(__file__).parent.parent.parent / "results" / "isa-optimal.csv",
            infer_schema_length=1000,
            null_values=["", "NA", "NaN", "null", "inf"]
        ).select(["instance", "rmp_opt"])

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

    # Farkas Initialization (sorted by custom algorithm order)

    # NOTE: for any 'instance', 'algorithm' that is missing 'last' == -1 or 6,
    farkas = df.filter(pl.col("last").is_in([-1, 6, -4])).filter(pl.col("rmp_opt").is_not_null())

    # cap rmptime at 3600 seconds for timeout entries
    farkas = farkas.with_columns([
        pl.when(pl.col("rmptime") > 3600)
          .then(3600)
          .otherwise(pl.col("rmptime"))
          .alias("rmptime_cap")
    ])

    farkas = (
        farkas
          .group_by(["algorithm"])
          .agg([
              avg_expr("rmptime_cap"),
              avg_expr("cgtime"),
              avg_expr("iterations"),
              avg_expr("has_integer"),
              (pl.col("time") >= 600).sum().alias(f"timeout"),
              pl.len().alias("count"),

              # gap relative to optimal integer solution
              gap_expr("obj"),
          ])
          .with_columns(
              pl.col("algorithm").replace_strict(_alg_order_map, default=999).alias("_alg_order")
          )
          .sort(["_alg_order"])
          .drop("_alg_order")
    )

    print(farkas)


    # capture cumulative count of solved vs time for each algorithm,
    # where time is quantized in minutes and cumulative count is number of instances solved within that time
    solved = df.filter(pl.col("last").is_in([1])).filter(pl.col("rmp_opt").is_not_null())

    # solved = solved.with_columns([
    #     pl.when(pl.col("time") > 3600)
    #       .then(3600*2)
    #       .otherwise(pl.col("time"))
    #       .alias("time_cap")
    # ])

    # print LT instances that don't time out, but also don't solve to optimality (rmp_opt != LB)
    print(solved.filter(pl.col("algorithm") == "LT")
          .filter(pl.col("timeout") == 0)
          .filter(pl.col("rmp_opt") != (pl.col("lb") - 1e-6).ceil()))


    # only consider instances solved to optimality (rmp_opt == LB)
    solved = solved.filter(((pl.col("algorithm") != "lr") | (pl.col("rmp_opt") <= (pl.col("lb") - 1e-6).ceil())))
    solved = solved.filter(pl.col("timeout") == 0)

    print(solved.group_by("algorithm").agg(pl.len().alias("count")))

    solved = solved.with_columns([
        pl.when(pl.col("time") > 3600)
          .then(3600)
          .otherwise(pl.col("time"))
          .alias("time_cap")
    ])

    # quantize time to minutes
    solved = solved.with_columns([
        ((pl.col("time_cap") // 30) / 2.0).cast(pl.Float64).alias("time_minutes")
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

    print(solved)

    # join with all_time_minutes to ensure no gaps, then forward fill
    solved = (all_time_minutes
        .join(solved, on="time_minutes", how="left")
        .sort("time_minutes")
        .fill_null(strategy="forward")
        .fill_null(0)  # fill any remaining nulls at the start with 0
        .filter(pl.col("time_minutes") <= 60))

    # convert cumsum to percentage, i.e., divide by 1735 instances
    solved = solved.with_columns([
        (pl.col("Wentges") / 1735 * 100).alias("Wentges"),
        (pl.col("LT") / 1735 * 100).alias("LT"),
        (pl.col("lr") / 1735 * 100).alias("LR"),
    ]).select(["time_minutes", "Wentges", "LT", "LR"])

    print(solved)
    solved.write_csv("agg_solved.csv")



    stats = df.filter(pl.col("last").is_in([1])).filter(pl.col("rmp_opt").is_not_null())

    stats = stats.with_columns([
        pl.when(pl.col("rmptime") > 3600)
          .then(3600)
          .otherwise(pl.col("rmptime"))
          .alias("rmptime_cap")
    ])

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
              (1735 - sum_expr("timeout")).alias("non-timeouts"),
              (1-sum_expr("timeout") / 1735).alias("solved"),
              ((pl.col("rmp_opt") <= (pl.col("lb") - 1e-6).ceil()) & (pl.col("timeout") == 0)).sum().alias("lr_treated_as_solved"),
              avg_expr("iterations"),
              avg_expr("rmptime_cap"),
              avg_expr("cgtime"),

              avg_expr("lpiters_per_column"),
              (sum_expr("lpiters") / sum_expr("rmptime")).alias("lpss"),
              (sum_expr("has_integer") / 1735).alias("integral"),
              avg_expr("gap"),
              pl.len().alias("count"),
              (pl.col("rmp_opt") <= (pl.col("lb") - 1e-6).ceil()).sum().alias("optimal")
          ])
          .with_columns(
              pl.col("algorithm").replace_strict(_alg_order_map, default=999).alias("_alg_order")
          )
          .sort(["_alg_order"])
          .drop(["_alg_order"])
    )

    print(stats)
    stats.write_csv("agg_stats.csv")


    # class/job status
    class_job = df.filter(pl.col("last").is_in([1])).filter(pl.col("rmp_opt").is_not_null())

    class_job = class_job.with_columns([
        pl.when(pl.col("rmptime") > 3600)
          .then(3600)
          .otherwise(pl.col("rmptime"))
          .alias("rmptime_cap")
    ])


    class_job = class_job.with_columns([
        pl.when(pl.col("rmp_opt") <= (pl.col("lb") - 1e-6).ceil())
          .then(1)
          .otherwise(0)
          .alias("hasopt")
    ])

    class_job = (
        class_job
         .group_by(["algorithm", "class", "jobs"]).agg([
            (avg_expr("hasopt") * 100).alias("solved"),
            ((pl.col("rmp_opt") <= (pl.col("lb") - 1e-6).ceil()) & (pl.col("timeout") == 0)).mean().alias("lr_treated_as_solved"),
            (avg_expr("timeout")).alias("timeouts"),
            avg_expr("rmptime_cap").alias("RMP"),
            avg_expr("cgtime").alias("CG"),
            pl.len().alias("count"),
            (avg_expr("has_integer")*100).alias("integral"),
        ])
         .with_columns(
             pl.col("algorithm").replace_strict(_alg_order_map, default=999).alias("_alg_order")
         )
         .sort("_alg_order", "class", "jobs")
         .drop("_alg_order")
    )

    # pivot by class and jobs, with columns for each algorithm, values are avg hasopt, runtime_cap, cgtime, has_integer
    class_job_pivot = class_job.pivot(
        values=["solved", "timeouts", "lr_treated_as_solved", "RMP", "CG", "integral", "count"],
        index=["class", "jobs"],
        on="algorithm"
    )

    print(class_job_pivot)
    class_job_pivot.write_csv("agg_class_job.csv")

    # try:
    #     import xlsxwriter

    #     # take all the csvs and write to a single excel file with multiple sheets
    #     with xlsxwriter.Workbook("agg_statistics.xlsx") as workbook:
    #         farkas.write_excel(workbook=workbook, worksheet="farkas", autofilter=False, autofit=True)
    #         column_retention.write_excel(workbook=workbook, worksheet="retention", autofilter=False, autofit=True)
    #         best.write_excel(workbook=workbook, worksheet="best", autofilter=False, autofit=True)
    #         degeneracy.write_excel(workbook=workbook, worksheet="degeneracy", autofilter=False, autofit=True)
    #         instances.write_excel(workbook=workbook, worksheet="instances", autofilter=False, autofit=True)
    #         solved_vs_time.write_excel(workbook=workbook,
    #                                    worksheet="solved_vs_time",
    #                                    autofilter=False,
    #                                    autofit=True)
    #         violin_data.write_excel(workbook=workbook, worksheet="violin_data", autofilter=False, autofit=True)

    # except:
    #     print("xlsxwriter not available, skipping Excel output")
    #     farkas.write_csv("agg_farkas.csv")
    #     column_retention.write_csv("agg_retention.csv")
    #     best.write_csv("agg_best.csv")
    #     degeneracy.write_csv("agg_degeneracy.csv")
    #     instances.write_csv("agg_instances.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())