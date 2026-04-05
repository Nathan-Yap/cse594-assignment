import os
import glob
import argparse
import pandas as pd


def combine_csvs(input_dir: str, output_file: str):
    """
    Combine all CSV files in input_dir into one CSV and deduplicate
    based on (session_id, image_name).
    """

    # Find all CSV files
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_dir}")

    print(f"Found {len(csv_files)} CSV files")

    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
            print(f"Loaded: {file} ({len(df)} rows)")
        except Exception as e:
            print(f"Skipping {file}: {e}")

    # Combine all rows
    combined = pd.concat(dataframes, ignore_index=True)

    print(f"\nTotal rows before deduplication: {len(combined)}")

    # Deduplicate based on session_id + image_name
    deduped = combined.drop_duplicates(
        subset=["session_id", "image_name"],
        keep="first"
    )

    print("Num unique ids:", len(set(deduped['session_id'])))

    print(f"Total rows after deduplication: {len(deduped)}")

    # Save to CSV
    deduped.to_csv(output_file, index=False)
    print(f"\nSaved combined file to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine survey CSV files and deduplicate by session_id + image_name"
    )
    parser.add_argument("input_dir", help="Directory containing CSV files")
    parser.add_argument(
        "-o",
        "--output",
        default="combined.csv",
        help="Output CSV file name"
    )

    args = parser.parse_args()

    combine_csvs(args.input_dir, args.output)