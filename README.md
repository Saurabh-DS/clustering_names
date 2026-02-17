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
- **Memory-efficient**: Processes only unique names (~250k), not all 30M rows. Final merge uses Polars.
- **Configurable threshold**: Tune `SIMILARITY_THRESHOLD` in `config.py` (default: 0.70). Lower = more aggressive merging, higher = more conservative.

## Cleaning Pipeline

1. **Date removal** — `DD/MM/YYYY`, `MM-DD-YY`, `YYYY-MM-DD`, `DD Mon YYYY`, etc.
2. **Prefix removal** — `Mr`, `Mrs`, `Miss`, `Ms`, `Dr`, `Moham` (whole-word only — won't break "Mistral" or "Drummond")
3. **Noise stripping** — Leading/trailing punctuation, special characters, digits
4. **Standardization** — Lowercase, collapse whitespace, strip
