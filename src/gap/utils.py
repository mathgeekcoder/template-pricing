#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import polars as pl

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_NULL_VALUES = ["", "NA", "NaN", "null", "inf"]

def normalize_algorithm_names(df: pl.DataFrame) -> pl.DataFrame:
    if "algorithm" not in df.columns:
        return df

    return df.with_columns([
        (pl.col("algorithm")
            .str.replace("Dantzig", "D")
            .str.replace("Wentges", "P")
            .str.replace("LagrangeTemplate", "LT")
            .str.replace("MipTemplate", "MT")
            .str.replace("lr", "LR")
        ).alias("algorithm")
    ])


def clean_gap_column(df: pl.DataFrame) -> pl.DataFrame:
    if "gap" not in df.columns:
        return df

    return df.with_columns([
        pl.col("gap").cast(pl.String)
        .str.replace_all(r"#NAME\?|nan|^$", "")
        .alias("gap")
    ]).with_columns([
        pl.when(pl.col("gap") == "")
        .then(None)
        .otherwise(pl.col("gap").cast(pl.Float64, strict=False))
        .alias("gap")
    ])


def load_results_csv(dataset: str, file: str) -> pl.DataFrame:
    try:
        df = pl.read_csv(
            PROJECT_ROOT / "results" / dataset / file,
            infer_schema_length=100000,
            null_values=DEFAULT_NULL_VALUES,
        )
    except Exception as e:
        raise Exception(f"Failed to read CSV: {e}")

    df = normalize_algorithm_names(df)
    return clean_gap_column(df)


def sort_by_algorithm(df: pl.DataFrame) -> pl.DataFrame:
    algorithm_order_map = {name: i for i, name in enumerate(["D", "P", "LT", "MT", "LR"])}

    return (
        df.with_columns(
            pl.col("algorithm").replace_strict(
                algorithm_order_map,
                default=999,
            ).alias("_alg_order")
        )
        .sort(["_alg_order"])
        .drop("_alg_order")
    )

def extract_parameters(df: pl.DataFrame) -> pl.DataFrame:
    if "params" not in df.columns:
        return df

    # extract "age_limit" and "max_col_multiplier" json fields from params column
    return df.with_columns([
        pl.col("params").str.extract(r".*age_limit:\s*(\d+)").cast(pl.Int64).alias("age_limit"),
        pl.col("params").str.extract(r".*random_seed:\s*(\d+)").cast(pl.Int64).alias("random_seed"),
        pl.col("params").str.extract(r".*max_col_multiplier:\s*(\d+)").cast(pl.Float64).alias("max_col_multiplier"),
        (pl.col("jobs").cast(pl.Int64) // pl.col("machines").cast(pl.Int64)).alias("job_machine_ratio")
    ])
