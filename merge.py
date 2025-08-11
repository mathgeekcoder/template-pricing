import polars as pl
import glob
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Merge all CSV files for a given path")
    parser.add_argument('pattern', nargs='?', default='*.csv', help="File pattern to match CSV files (default: '*.csv')")
    parser.add_argument('-o', '--output', default='join.csv', help="Output filename (default: 'join.csv')")
    args = parser.parse_args()

    if os.path.exists(args.output):
        confirm = input(f"Output file {args.output} already exists. Overwrite? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Aborted. Output file not overwritten.")
            return

    csv_files = glob.glob(args.pattern)
    frames = []

    for f in csv_files:
        if f == args.output:
            print(f"Skipping output file: {args.output}")
            continue

        df = pl.read_csv(f)
        frames.append(df)
        print(f"Merged: {os.path.basename(f)}")

    if frames:
        try:
            result = pl.concat(frames)
            result.write_csv(args.output)
            print(f"All files merged into {args.output}")
        except Exception as e:
            print(f"Error during concatenation: {e}")
    else:
        print("No CSV files found matching the pattern.")

if __name__ == "__main__":
    main()
