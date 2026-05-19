"""Merge raw LS-DYNA CSV exports into one training dataset.

The script expects a directory containing compatible CSV files. It concatenates
all rows, removes fully empty rows, inserts a numeric index column, and writes a
single cleaned CSV file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def merge_csv_files(input_dir: Path, output_path: Path) -> pd.DataFrame:
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    frames = []
    for csv_file in csv_files:
        frame = pd.read_csv(csv_file, index_col=0)
        frames.append(frame)

    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged = merged.dropna(axis=0, how="all")
    merged.insert(0, "index", np.arange(len(merged), dtype=float))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge raw LS-DYNA shear-wall CSV exports into one dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("raw_data"),
        help="Directory containing raw CSV files. Default: raw_data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("all_data.csv"),
        help="Merged CSV output path. Default: all_data.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged = merge_csv_files(args.input_dir, args.output)
    print(f"Merged {len(merged)} rows into {args.output}")


if __name__ == "__main__":
    main()
