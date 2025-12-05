#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
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


    # best filter
    best_retention_filter = (
        ((pl.col("algorithm") == "Dantzig") & (pl.col("retention") == "high")) |
        ((pl.col("algorithm") == "Wentges") & (pl.col("retention") == "med")) |
        ((pl.col("algorithm") == "MT") & (pl.col("retention") == "low")) |
        ((pl.col("algorithm") == "LT") & (pl.col("retention") == "low"))
    )

    # Farkas Initialization (sorted by custom algorithm order)
    farkas = (
        df.filter(best_retention_filter).filter(pl.col("last") == -1)
          .group_by(["algorithm"])
          .agg([
              avg_expr("rmptime"),
              avg_expr("cgtime"),
              avg_expr("iterations"),
              avg_expr("has_integer"),

              # gap relative to optimal integer solution
              gap_expr("obj"),

              geo_expr("rmptime") * 1000, # convert to milliseconds
              geo_expr("cgtime") * 1000,  # convert to milliseconds
              geo_expr("iterations"),
          ])
          .with_columns(
              pl.col("algorithm").replace_strict(_alg_order_map, default=999).alias("_alg_order")
          )
          .sort(["_alg_order"])
          .drop("_alg_order")
    )

    column_retention = (
        df.filter(pl.col("last") == 1)
          .group_by(["algorithm", "retention"], maintain_order=True)
          .agg([
              avg_expr("rmptime"),
              avg_expr("cgtime"),
              avg_expr("lpiters_per_second"),
              avg_expr("lpiters_per_column"),

              geo_expr("rmptime"),
              geo_expr("cgtime"),
              geo_expr("lpiters_per_second"),
              geo_expr("lpiters_per_column")
          ]
            + boxplot_expr("rmptime")
            + boxplot_expr("cgtime")
            + boxplot_expr("lpiters_per_second")
            + boxplot_expr("lpiters_per_column")
          )
          .with_columns(
              pl.col("algorithm").replace_strict(_alg_order_map, default=999).alias("_alg_order"),
              pl.col("retention").replace_strict({"low": 0, "med": 1, "high": 2}, default=999).alias("_ret_order")
          )
          .sort(["_alg_order", "_ret_order"])
          .drop(["_alg_order", "_ret_order"])
    )

    # Best
    best = (
        df.filter(best_retention_filter)
         .filter(pl.col("last") == 1)
         .group_by(["algorithm", "class"]).agg([
            avg_expr("timeout"),
            avg_expr("iterations"),
            avg_expr("rmptime"),
            avg_expr("cgtime"),
            avg_expr("has_integer"),
            avg_expr("time"),

            geo_expr("time"),
            geo_expr("timeout"),
            geo_expr("iterations"),
            geo_expr("rmptime"),
            geo_expr("cgtime"),

            # gap relative to optimal integer solution
            gap_expr("ub"),
        ])
         .with_columns(
             pl.col("algorithm").replace_strict(_alg_order_map, default=999).alias("_alg_order")
         )
         .sort("_alg_order")
         .drop("_alg_order")
    )

    # Degeneracy
    degeneracy = (
        df.filter(best_retention_filter)
         .filter(pl.col("last") == 1)
         .group_by(["algorithm","jobs","machines"]).agg([
            avg_expr("rmptime"),
            avg_expr("cgtime"),
            geo_expr("rmptime"),
            geo_expr("cgtime"),
        ])
         .with_columns(
             pl.col("algorithm").replace_strict(_alg_order_map, default=999).alias("_alg_order")
         )
         .sort(["_alg_order","jobs","machines"])
         .drop("_alg_order")
    )

    # instances
    instances = (
        df.filter(best_retention_filter)
         .filter(pl.col("last") == 1)
         .group_by(["algorithm","class","jobs","machines"]).agg([
            avg_expr("rmptime"),
            avg_expr("cgtime"),
            geo_expr("rmptime"),
            geo_expr("cgtime"),
        ])
         .with_columns(
             pl.col("algorithm").replace_strict(_alg_order_map, default=999).alias("_alg_order")
         )
         .sort(["_alg_order","class","jobs","machines"])
         .drop("_alg_order")
    )


    # number solved vs time to solve
    # want "method,count,time_to_solve"
    # where count is cumulative number of instances solved by time_to_solve,
    # and time_to_solve is quantized in a log scale (e.g., seconds, minutes, hours)
    solved_vs_time = (df.filter(best_retention_filter)
                        .filter(pl.col("last") == 1)
                        .with_columns([(pl.col("time") // 1).alias("time_to_solve")])
    )

    all_times = (solved_vs_time
                 .select(pl.col("time_to_solve"))
                 .unique()
                 .sort("time_to_solve")['time_to_solve'].to_list())

    # create a complete set of time_to_solve values for each algorithm
    solved_vs_time = (solved_vs_time
        .group_by(['algorithm',"time_to_solve"])
            .agg([pl.col("instance").count().alias("count")])
        .sort(['algorithm',"time_to_solve"])
        .with_columns(
            pl.col("count").cum_sum().over('algorithm').alias("cumsum")
        )
        .pivot(values="cumsum", index="time_to_solve", columns="algorithm")
        .sort("time_to_solve")
        .fill_null(strategy='forward')
    )


    # violin plot data for time to solve by algorithm
    violin_data = (df
        .filter(best_retention_filter)
        .filter(pl.col("last") == 1)
        # add column with instance-replication to ensure uniqueness
        .with_columns(
            (pl.col("instance") + "_" + pl.col("replication").cast(pl.Utf8)).alias("unique_instance")
        )
        .pivot(values="time", index="unique_instance", columns="algorithm")
    )

    # print results
    print(farkas)
    print(column_retention)
    print(best)
    print(degeneracy)
    print(instances)
    print(solved_vs_time)
    print(violin_data)

    try:
        import xlsxwriter

        # take all the csvs and write to a single excel file with multiple sheets
        with xlsxwriter.Workbook("agg_statistics.xlsx") as workbook:
            farkas.write_excel(workbook=workbook, worksheet="farkas", autofilter=False, autofit=True)
            column_retention.write_excel(workbook=workbook, worksheet="retention", autofilter=False, autofit=True)
            best.write_excel(workbook=workbook, worksheet="best", autofilter=False, autofit=True)
            degeneracy.write_excel(workbook=workbook, worksheet="degeneracy", autofilter=False, autofit=True)
            instances.write_excel(workbook=workbook, worksheet="instances", autofilter=False, autofit=True)
            solved_vs_time.write_excel(workbook=workbook,
                                       worksheet="solved_vs_time",
                                       autofilter=False,
                                       autofit=True)
            violin_data.write_excel(workbook=workbook, worksheet="violin_data", autofilter=False, autofit=True)

    except:
        print("xlsxwriter not available, skipping Excel output")
        farkas.write_csv("agg_farkas.csv")
        column_retention.write_csv("agg_retention.csv")
        best.write_csv("agg_best.csv")
        degeneracy.write_csv("agg_degeneracy.csv")
        instances.write_csv("agg_instances.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())