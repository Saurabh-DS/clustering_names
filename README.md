# PayeeName Entity Resolution Pipeline

A high-performance Python pipeline that cleans and clusters **30M+ rows** of manually-entered payee names into standardized groups using NLP and sparse matrix operations.

## Architecture

```
30M raw rows → Extract 250k unique names → Regex Cleaning → TF-IDF Char N-Grams
    → Sparse Cosine Similarity (sparse_dot_topn) → Connected Components → Cluster IDs
        → Representative Name Selection → Merge back to 30M rows
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data + run the full pipeline (small test)
python pipeline.py --generate --num-unique 500 --num-rows 10000

# Production-scale run
python pipeline.py --generate --num-unique 250000 --num-rows 30000000

# Run on your own data (CSV must have a 'PayeeName' column)
python pipeline.py --input your_data.csv --output-dir ./output
```

## Project Structure

| File                         | Purpose                                                    |
| ---------------------------- | ---------------------------------------------------------- |
| `config.py`                  | Central configuration (paths, thresholds, n-gram settings) |
| `generate_synthetic_data.py` | Creates realistic messy payee names for testing            |
| `cleaning.py`                | Regex-based preprocessing (prefix, date, noise removal)    |
| `clustering.py`              | TF-IDF + sparse cosine similarity + connected components   |
| `pipeline.py`                | Main orchestrator — ties everything together               |

## Output

| File                           | Description                                                       |
| ------------------------------ | ----------------------------------------------------------------- |
| `output/mapping_table.csv`     | `original_name → cleaned_name → cluster_id → representative_name` |
| `output/final_dataset.parquet` | Full dataset with cleaning/clustering columns merged back         |

## Performance Notes

- **No nested loops**: Uses sparse matrix multiplication (`sparse_dot_topn`) instead of comparing every pair — handles 250k names in minutes.
- **Memory-efficient**: Processes only unique names (~250k), not all 30M rows. Final merge uses Polars (local) or Spark broadcast join (Databricks).
- **Configurable threshold**: Tune `SIMILARITY_THRESHOLD` in `config.py` (default: 0.70). Lower = more aggressive merging, higher = more conservative.

## Running on Databricks

The `pipeline_databricks.py` file is a **Databricks notebook** (`.py` format with `# COMMAND ----------` cells). To use it:

1. **Import** the file into your Databricks workspace (`Workspace > Import > File`)
2. **Update the config** in the first code cell:
   ```python
   INPUT_TABLE = "your_catalog.your_schema.your_table"  # Delta table with PayeeName column
   PAYEE_COLUMN = "PayeeName"                           # column name in your table
   OUTPUT_MAPPING_TABLE = "your_catalog.your_schema.payee_mapping"
   OUTPUT_FINAL_TABLE   = "your_catalog.your_schema.payee_resolved"
   ```
3. **Run all cells** — the notebook handles `%pip install` and library restart automatically.

### How it works on Databricks

| Step                  | Engine                   | Why                                               |
| --------------------- | ------------------------ | ------------------------------------------------- |
| Read 30M rows         | **Spark**                | Distributed read from Delta                       |
| Extract unique names  | **Spark** `.distinct()`  | Parallel deduplication                            |
| Clean 250k names      | **Pandas** (driver)      | Regex is single-threaded, 250k fits in memory     |
| Cluster 250k names    | **Pandas** (driver)      | scikit-learn TF-IDF + sparse_dot_topn             |
| Join back to 30M rows | **Spark broadcast join** | Mapping table (~250k) is broadcast to all workers |
| Write results         | **Spark**                | Distributed write to Delta                        |

### Recommended cluster spec

- **Driver**: `Standard_DS4_v2` or similar (32 GB RAM) — requires enough memory for 250k names in pandas
- **Workers**: 2-4 nodes for the 30M-row join

## Cleaning Pipeline

1. **Date removal** — `DD/MM/YYYY`, `MM-DD-YY`, `YYYY-MM-DD`, `DD Mon YYYY`, etc.
2. **Prefix removal** — `Mr`, `Mrs`, `Miss`, `Ms`, `Dr`, `Moham` (whole-word only — won't break "Mistral" or "Drummond")
3. **Noise stripping** — Leading/trailing punctuation, special characters, digits
4. **Standardization** — Lowercase, collapse whitespace, strip
