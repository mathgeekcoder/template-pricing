import polars as pl
import glob
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Merge all CSV files for a given path")
    parser.add_argument('pattern', nargs='+', default='*.csv', help="File pattern to match CSV files (default: '*.csv')")
    parser.add_argument('-o', '--output', default='join.csv', help="Output filename (default: 'join.csv')")
    args = parser.parse_args()

    output_path = os.path.abspath(args.output)

    if os.path.exists(output_path):
        confirm = input(f"Output file {args.output} already exists. Overwrite? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Aborted. Output file not overwritten.")
            return

    csv_files = [os.path.abspath(f) for p in args.pattern for f in glob.glob(p)]
    frames = []

    for f in csv_files:
        if f == output_path:
            print(f"Skipping output file: {os.path.basename(args.output)}")
            continue

        df = pl.read_csv(f, infer_schema=False)
        frames.append(df)
        print(f"Loaded: {os.path.basename(f)}")

    if frames:
        try:
            result = pl.concat(frames)
            result.write_csv(output_path)
            print(f"All files merged into {args.output}")
        except Exception as e:
            print(f"Error during concatenation: {e}")
    else:
        print("No CSV files found matching the pattern.")

if __name__ == "__main__":
    main()
