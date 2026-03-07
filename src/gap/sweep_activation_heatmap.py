from utils import load_results_csv, extract_parameters
import polars as pl

def render_cli_heatmap(pivot: pl.DataFrame) -> None:
    """
    Render a compact ANSI heatmap in the CLI.
    Expects:
      - row index column: max_col_multiplier
      - all other columns: age_limit values
      - cell values: normalized_time in [0, 1]
    """
    age_columns = [c for c in pivot.columns if c != "max_col_multiplier"]

    # green -> yellow -> red (ANSI 256-color palette)
    palette = [34, 40, 46, 82, 118, 154, 190, 220, 208, 196]

    def cell_for(value: float | None) -> str:
        if value is None:
            return "  "
        idx = int(max(0.0, min(1.0, float(value))) * (len(palette) - 1))
        color = palette[idx]
        return f"\x1b[48;5;{color}m  \x1b[0m"

    print(f"\nCLI heatmap (normalized_time)  fast \x1b[48;5;{palette[0]}m  \x1b[0m ... \x1b[48;5;{palette[-1]}m  \x1b[0m slow\n")

    header = "mult\\age | " + " ".join(f"{str(a):>2}" for i, a in enumerate(age_columns) if i % 2 == 0)
    print(header)
    print("-" * len(header))

    for row in pivot.iter_rows(named=True):
        mult = row["max_col_multiplier"]
        line = f"{mult:>8.0f} | " + "".join(cell_for(row[a]) for a in age_columns)
        print(line)


if __name__ == "__main__":
    pl.Config.set_tbl_cell_alignment("RIGHT")
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    instances = load_results_csv("parameter-sweeps", "c10400-age-vs-activation-sweep.csv")
    instances = extract_parameters(instances).filter(pl.col("last") == 1)

    # too much data to plot all, so group age into buckets of size 20 and max_col_multiplier into buckets of size 2
    instances = instances.with_columns([
        (pl.col("age_limit") // 20 * 20 + 20).alias("age_limit"),
        (pl.col("max_col_multiplier") // 2 * 2 + 1).alias("max_col_multiplier")
    ])

    instances = instances.group_by([
        "instance", "age_limit", 'max_col_multiplier'
    ]).agg([pl.col("time").mean().alias("mean_time")])

    min_time = instances['mean_time'].min()
    max_time = instances.filter(pl.col('mean_time') < 3600)['mean_time'].max()  # exclude timeout

    # normalize mean_time to [0, 1] range
    instances = instances.with_columns([
        pl.min_horizontal(1, ((pl.col("mean_time") - min_time) /
         (max_time - min_time))).alias("normalized_time")
    ])

    instances = instances.sort(['instance', 'age_limit', 'max_col_multiplier'])

    # pivot age to columns and max_col_multiplier to rows, with values as normalized_time
    pivot = instances.pivot(values="normalized_time",
                            index="max_col_multiplier",
                            on="age_limit")

    print(pivot)
    render_cli_heatmap(pivot)
    instances.write_csv("yagiura-c10400-age-vs-activation-heatmap.csv")
