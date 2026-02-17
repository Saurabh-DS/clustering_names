"""
PayeeName Entity Resolution Pipeline
======================================
Main orchestrator that ties together:
  1. Data loading
  2. Unique name extraction
  3. Cleaning
  4. Clustering
  5. Mapping back to the full dataset
  6. Export

Usage:
    python pipeline.py --input data/synthetic_payees.csv --output-dir output/
    python pipeline.py --generate --num-unique 500 --num-rows 10000  # generate + run
"""
import argparse
import time
from pathlib import Path

import pandas as pd

import config
from cleaning import clean_names
from clustering import cluster_names


def load_data(input_path: Path) -> pd.DataFrame:
    """Load the raw dataset."""
    print(f"\n[1/7] Loading data from {input_path}...")
    t0 = time.time()
    df = pd.read_csv(input_path, dtype={"PayeeName": str})
    elapsed = time.time() - t0
    print(f"  Loaded {len(df):,} rows in {elapsed:.1f}s")
    print(f"  Columns: {list(df.columns)}")
    return df


def extract_unique_names(df: pd.DataFrame) -> pd.Series:
    """Extract unique PayeeName values."""
    print(f"\n[2/7] Extracting unique names...")
    unique_names = df["PayeeName"].dropna().drop_duplicates().reset_index(drop=True)
    print(f"  Total rows:    {len(df):,}")
    print(f"  Unique names:  {len(unique_names):,}")
    print(f"  Reduction:     {(1 - len(unique_names)/len(df))*100:.2f}%")
    return unique_names


def clean_unique_names(unique_names: pd.Series) -> pd.DataFrame:
    """Clean unique names and return a mapping."""
    print(f"\n[3/7] Cleaning {len(unique_names):,} unique names...")
    t0 = time.time()
    cleaned = clean_names(unique_names)
    elapsed = time.time() - t0

    cleaning_map = pd.DataFrame({
        "original_name": unique_names.values,
        "cleaned_name": cleaned.values,
    })

    # Stats
    changed = (cleaning_map["original_name"].str.lower().str.strip()
               != cleaning_map["cleaned_name"])
    print(f"  Cleaned in {elapsed:.1f}s")
    print(f"  Names modified: {changed.sum():,} / {len(cleaning_map):,} "
          f"({changed.mean()*100:.1f}%)")
    print(f"  Unique after cleaning: {cleaning_map['cleaned_name'].nunique():,}")

    return cleaning_map


def cluster_cleaned_names(cleaning_map: pd.DataFrame) -> pd.DataFrame:
    """Cluster the cleaned names and add cluster info to the mapping."""
    print(f"\n[4/7] Clustering cleaned names...")
    t0 = time.time()

    unique_cleaned = pd.Series(
        cleaning_map["cleaned_name"].unique(),
        name="cleaned_name",
    )

    cluster_result = cluster_names(unique_cleaned)
    elapsed = time.time() - t0

    # Merge cluster info back to the cleaning map
    full_map = cleaning_map.merge(cluster_result, on="cleaned_name", how="left")

    print(f"\n  Clustering completed in {elapsed:.1f}s")
    return full_map


def build_mapping_table(full_map: pd.DataFrame) -> pd.DataFrame:
    """Finalize the mapping table."""
    print(f"\n[5/7] Building mapping table...")
    mapping = full_map[["original_name", "cleaned_name", "cluster_id", "representative_name"]].copy()
    mapping = mapping.drop_duplicates(subset=["original_name"])

    print(f"  Mapping entries: {len(mapping):,}")
    print(f"  Clusters:        {mapping['cluster_id'].nunique():,}")
    print(f"  Representatives: {mapping['representative_name'].nunique():,}")

    # Show some sample clusters
    print(f"\n  Sample clusters (top 5 largest):")
    top_clusters = mapping["cluster_id"].value_counts().head(5).index
    for cid in top_clusters:
        members = mapping[mapping["cluster_id"] == cid]
        rep = members["representative_name"].iloc[0]
        sample_originals = members["original_name"].head(5).tolist()
        print(f"    Cluster {cid} -> \"{rep}\" ({len(members)} members)")
        for orig in sample_originals:
            print(f"      - \"{orig}\"")

    return mapping


def merge_back(df: pd.DataFrame, mapping: pd.DataFrame, use_polars: bool = None) -> pd.DataFrame:
    """Merge the mapping table back to the full dataset."""
    use_polars = use_polars if use_polars is not None else config.USE_POLARS_FOR_MERGE

    print(f"\n[6/7] Merging mapping back to {len(df):,} rows...")
    t0 = time.time()

    if use_polars:
        try:
            import polars as pl
            print("  Using Polars for memory-efficient merge...")

            df_pl = pl.from_pandas(df)
            mapping_pl = pl.from_pandas(
                mapping[["original_name", "cleaned_name", "cluster_id", "representative_name"]]
            )

            result_pl = df_pl.join(
                mapping_pl,
                left_on="PayeeName",
                right_on="original_name",
                how="left",
            )

            result = result_pl.to_pandas()
        except ImportError:
            print("  Polars not available, falling back to pandas...")
            result = df.merge(
                mapping[["original_name", "cleaned_name", "cluster_id", "representative_name"]],
                left_on="PayeeName",
                right_on="original_name",
                how="left",
            )
    else:
        result = df.merge(
            mapping[["original_name", "cleaned_name", "cluster_id", "representative_name"]],
            left_on="PayeeName",
            right_on="original_name",
            how="left",
        )

    elapsed = time.time() - t0
    print(f"  Merged in {elapsed:.1f}s")
    print(f"  Result shape: {result.shape}")
    return result


def export_results(
    mapping: pd.DataFrame,
    final_df: pd.DataFrame,
    output_dir: Path,
):
    """Export the mapping table and final dataset."""
    print(f"\n[7/7] Exporting results to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = output_dir / "mapping_table.csv"
    final_path = output_dir / "final_dataset.csv"

    mapping.to_csv(mapping_path, index=False)
    print(f"  Mapping table: {mapping_path}  ({len(mapping):,} rows)")

    final_df.to_csv(final_path, index=False)
    print(f"  Final dataset: {final_path}  ({len(final_df):,} rows)")

    return mapping_path, final_path


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(
    input_path: Path = None,
    output_dir: Path = None,
    generate: bool = False,
    num_unique: int = None,
    num_rows: int = None,
):
    """
    Run the full entity resolution pipeline.

    Parameters
    ----------
    input_path : Path
        Path to the input CSV with a 'PayeeName' column.
    output_dir : Path
        Directory for output files.
    generate : bool
        If True, generate synthetic data first.
    num_unique : int
        Number of unique names for synthetic generation.
    num_rows : int
        Total rows for synthetic generation.
    """
    input_path = input_path or config.SYNTHETIC_DATA_PATH
    output_dir = output_dir or config.OUTPUT_DIR

    print("\n" + "=" * 60)
    print("  PayeeName Entity Resolution Pipeline")
    print("=" * 60)

    pipeline_start = time.time()

    # Step 0: Optionally generate synthetic data
    if generate:
        from generate_synthetic_data import generate_dataset
        input_path, _ = generate_dataset(
            num_unique=num_unique,
            num_total_rows=num_rows,
            output_path=input_path,
        )

    # Step 1: Load
    df = load_data(input_path)

    # Step 2: Extract unique
    unique_names = extract_unique_names(df)

    # Step 3: Clean
    cleaning_map = clean_unique_names(unique_names)

    # Step 4: Cluster
    full_map = cluster_cleaned_names(cleaning_map)

    # Step 5: Build mapping table
    mapping = build_mapping_table(full_map)

    # Step 6: Merge back
    final_df = merge_back(df, mapping)

    # Step 7: Export
    mapping_path, final_path = export_results(mapping, final_df, output_dir)

    total_time = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f"  Pipeline completed in {total_time:.1f}s")
    print(f"  Mapping:  {mapping_path}")
    print(f"  Dataset:  {final_path}")
    print(f"{'='*60}\n")

    return mapping, final_df


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PayeeName Entity Resolution Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on existing data
  python pipeline.py --input data/synthetic_payees.csv

  # Generate synthetic data and run pipeline
  python pipeline.py --generate --num-unique 500 --num-rows 10000

  # Full production run
  python pipeline.py --generate --num-unique 250000 --num-rows 30000000
        """,
    )
    parser.add_argument("--input", type=str, default=str(config.SYNTHETIC_DATA_PATH),
                        help="Path to input CSV with PayeeName column")
    parser.add_argument("--output-dir", type=str, default=str(config.OUTPUT_DIR),
                        help="Output directory for results")
    parser.add_argument("--generate", action="store_true",
                        help="Generate synthetic data before running")
    parser.add_argument("--num-unique", type=int, default=config.NUM_UNIQUE_NAMES,
                        help="Number of unique variants (for --generate)")
    parser.add_argument("--num-rows", type=int, default=config.NUM_TOTAL_ROWS,
                        help="Total rows (for --generate)")

    args = parser.parse_args()

    run_pipeline(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        generate=args.generate,
        num_unique=args.num_unique,
        num_rows=args.num_rows,
    )
