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
   "params",
   "last",
]

def geo_expr(column: str) -> pl.Expr:
    """
    Build a Polars expression computing geometric mean of positive values in 'column'.
    Excludes non-positive entries.
    """
    col = pl.col(column)
    return col.filter(col > 0).log().mean().exp().alias(f"geo_{column}")


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
            infer_schema_length=10000,
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
        pl.col("params").str.extract(r".*max_col_multiplier:\s*(\d+)").cast(pl.Float64).alias("max_col_multiplier")
    ])

    df = df.with_columns(
        [(pl.col("jobs").cast(pl.Int64) // pl.col("machines").cast(pl.Int64)).alias("job_machine_ratio")]
    )

    # instances
    # instances = (
    #     df.filter(pl.col("last") == 1)
    #       .filter(pl.col("max_col_multiplier") == 1)
    #       .group_by(["algorithm","class","jobs","machines","job_machine_ratio","age_limit"]).agg([
    #         geo_expr("rmptime")
    #     ])
    #     .filter(
    #         pl.col("geo_rmptime") == pl.col("geo_rmptime").min().over(["class","jobs","machines"])
    #     )
    #     .sort([
    #         pl.col("job_machine_ratio"),
    #         pl.col("geo_rmptime")
    #     ])
    # )

    # want to calculate a measure of "work" that 

    # TODO: want to take minimum time, but also age_limit
    # try to round the time to 1 decimal place to avoid numerical issues
    df = df.with_columns(
        (pl.col("time") * 1).round().alias("time_rounded"),
        (pl.col("lpiters_per_column") * pl.col("machines") * pl.col("iterations")).alias("work")
    )


    instances = (df.filter(pl.col("last") == 1).filter(
        pl.col("max_col_multiplier") == 1).filter(
            pl.col("timeout") == 0).filter(
                pl.col("work") == pl.col("work").min().over(
                    ["class", "jobs", "machines", "replication"])).select([
                        pl.col("algorithm"),
                        pl.col("class"),
                        pl.col("jobs"),
                        pl.col("machines"),
                        pl.col("replication"),
                        pl.col("job_machine_ratio"),
                        pl.col("age_limit"),
                        pl.col("work"),
                        pl.col("time")
                    ]).group_by([
                        "algorithm", "class", "jobs", "machines", "job_machine_ratio"
                    ]).agg([
                         pl.col("age_limit").min(),
                         pl.col("work").min(),
                         pl.col("time").min()
                        ]))

    print(instances)
    instances.write_csv("sweep_statistics-lagrange.csv")

    # try:
    #     import xlsxwriter

    #     # take all the csvs and write to a single excel file with multiple sheets
    #     with xlsxwriter.Workbook("agg_statistics.xlsx") as workbook:
    #         farkas.write_excel(workbook=workbook, worksheet="farkas", autofilter=False, autofit=True)
    #         column_retention.write_excel(workbook=workbook, worksheet="retention", autofilter=False, autofit=True)
    #         best.write_excel(workbook=workbook, worksheet="best", autofilter=False, autofit=True)
    #         degeneracy.write_excel(workbook=workbook, worksheet="degeneracy", autofilter=False, autofit=True)
    #         instances.write_excel(workbook=workbook, worksheet="instances", autofilter=False, autofit=True)
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