# Databricks notebook source
# MAGIC %md
# MAGIC # PayeeName Entity Resolution Pipeline — Databricks
# MAGIC
# MAGIC **Architecture:** Read from Delta table -> Extract unique names (Spark) ->
# MAGIC Clean + Cluster (Pandas on driver) -> Join back (Spark broadcast) -> Write to Delta
# MAGIC
# MAGIC The cleaning & clustering operate on ~250k unique names (fits in driver memory).
# MAGIC The 30M-row join uses Spark's distributed engine.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Configuration

# COMMAND ----------

# ─── Configuration ────────────────────────────────────────────────────────────
# Update these to match your Databricks environment

# Input: the table/view containing your 30M rows
INPUT_TABLE = "default.payee_raw"           # e.g. "catalog.schema.table"
PAYEE_COLUMN = "PayeeName"                  # column containing the payee names

# Output: where to write the results
OUTPUT_MAPPING_TABLE = "default.payee_mapping"       # cleaned name mapping
OUTPUT_FINAL_TABLE   = "default.payee_resolved"      # full dataset with clusters

# Clustering parameters
SIMILARITY_THRESHOLD = 0.70    # cosine similarity cutoff
NGRAM_RANGE = (2, 4)           # character n-gram window for TF-IDF
TOP_N_PER_ROW = 10             # max neighbours kept per name in similarity graph

# Prefixes to remove (whole-word, case-insensitive)
PREFIXES_TO_REMOVE = ["mr", "mrs", "miss", "ms", "dr", "moham"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies (run once per cluster)

# COMMAND ----------

# MAGIC %pip install sparse_dot_topn rapidfuzz --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Imports

# COMMAND ----------

import re
import time
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn
from scipy.sparse.csgraph import connected_components

# Re-import config after library restart
INPUT_TABLE = "default.payee_raw"
PAYEE_COLUMN = "PayeeName"
OUTPUT_MAPPING_TABLE = "default.payee_mapping"
OUTPUT_FINAL_TABLE   = "default.payee_resolved"
SIMILARITY_THRESHOLD = 0.70
NGRAM_RANGE = (2, 4)
TOP_N_PER_ROW = 10
PREFIXES_TO_REMOVE = ["mr", "mrs", "miss", "ms", "dr", "moham"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Data from Delta Table

# COMMAND ----------

print("=" * 60)
print("  PayeeName Entity Resolution Pipeline (Databricks)")
print("=" * 60)

pipeline_start = time.time()

# Step 1: Read the full dataset from Delta
print(f"\n[1/7] Loading data from {INPUT_TABLE}...")
t0 = time.time()
df_spark = spark.table(INPUT_TABLE)
total_rows = df_spark.count()
print(f"  Loaded {total_rows:,} rows in {time.time() - t0:.1f}s")
print(f"  Columns: {df_spark.columns}")

# Cache the source table for the join step later
df_spark.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Extract Unique Names (Spark)

# COMMAND ----------

print(f"\n[2/7] Extracting unique names using Spark...")
t0 = time.time()

unique_spark = (
    df_spark
    .select(PAYEE_COLUMN)
    .filter(F.col(PAYEE_COLUMN).isNotNull())
    .distinct()
)
unique_count = unique_spark.count()

print(f"  Total rows:   {total_rows:,}")
print(f"  Unique names: {unique_count:,}")
print(f"  Reduction:    {(1 - unique_count / total_rows) * 100:.2f}%")
print(f"  Extracted in {time.time() - t0:.1f}s")

# Convert to pandas — 250k strings fits easily in driver memory
unique_names_pd = unique_spark.toPandas()[PAYEE_COLUMN]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Clean Names (Pandas on Driver)

# COMMAND ----------

print(f"\n[3/7] Cleaning {len(unique_names_pd):,} unique names...")
t0 = time.time()

# ─── Compiled Regex Patterns ─────────────────────────────────────────────────
_PREFIX_WORDS = "|".join(re.escape(p) for p in PREFIXES_TO_REMOVE)
RE_PREFIXES = re.compile(rf"\b({_PREFIX_WORDS})\b\.?\s*", re.IGNORECASE)

RE_DATES = re.compile(
    r"""
    (?:
        \d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}
      | \d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}
      | \d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*(?:\s+\d{2,4})?
      | (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s*\d{0,4}
      | \d{8}
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Junk words: bacs, micl, postcodeservices, ltd, limited
RE_JUNK_WORDS = re.compile(
    r"\b(bacs|micl|postcodeservices|postcode\s*services|ltd|limited)\b\.?\s*",
    re.IGNORECASE,
)

RE_BRACKETS = re.compile(r"[(){}\[\]]")
RE_NUMBERS = re.compile(r"\d+")
RE_LEADING_NOISE = re.compile(r"^[^a-zA-Z]+")
RE_TRAILING_NOISE = re.compile(r"[^a-zA-Z]+$")
RE_MULTI_SPACE = re.compile(r"\s{2,}")


def clean_single(text):
    """Full cleaning pipeline for a single name string."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = RE_DATES.sub("", text)          # 1. dates
    text = RE_PREFIXES.sub("", text)       # 2. prefixes (Mr, Mrs, Dr, etc.)
    text = RE_JUNK_WORDS.sub("", text)     # 3. junk words (bacs, micl, ltd, etc.)
    text = RE_BRACKETS.sub("", text)       # 4. brackets
    text = RE_NUMBERS.sub("", text)        # 5. all numbers
    text = RE_LEADING_NOISE.sub("", text)  # 6. leading noise
    text = RE_TRAILING_NOISE.sub("", text) # 6. trailing noise
    text = text.lower()                    # 7. lowercase
    text = RE_MULTI_SPACE.sub(" ", text)   # 7. collapse spaces
    return text.strip()


cleaned_series = unique_names_pd.fillna("").astype(str).map(clean_single)

cleaning_map = pd.DataFrame({
    "original_name": unique_names_pd.values,
    "cleaned_name": cleaned_series.values,
})

changed = (cleaning_map["original_name"].str.lower().str.strip()
           != cleaning_map["cleaned_name"])
elapsed = time.time() - t0
print(f"  Cleaned in {elapsed:.1f}s")
print(f"  Names modified: {changed.sum():,} / {len(cleaning_map):,} ({changed.mean()*100:.1f}%)")
print(f"  Unique after cleaning: {cleaning_map['cleaned_name'].nunique():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Cluster Names (TF-IDF + Sparse Cosine Similarity)

# COMMAND ----------

print(f"\n[4/7] Clustering cleaned names...")
t0 = time.time()

# Get unique cleaned names
unique_cleaned = cleaning_map["cleaned_name"].unique()
unique_cleaned = pd.Series([n for n in unique_cleaned if n.strip()], name="cleaned_name")
n = len(unique_cleaned)

print(f"  Vectorizing {n:,} unique cleaned names...")

# TF-IDF with character n-grams
vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=NGRAM_RANGE,
    min_df=1,
    max_df=0.95,
    sublinear_tf=True,
    dtype=np.float32,
)
tfidf_matrix = vectorizer.fit_transform(unique_cleaned)
print(f"  TF-IDF matrix shape: {tfidf_matrix.shape} (nnz: {tfidf_matrix.nnz:,})")

# Sparse cosine similarity
print(f"  Computing sparse cosine similarity (threshold={SIMILARITY_THRESHOLD})...")
similarity = awesome_cossim_topn(
    tfidf_matrix,
    tfidf_matrix.T,
    ntop=TOP_N_PER_ROW,
    lower_bound=SIMILARITY_THRESHOLD,
    use_threads=True,
    n_jobs=4,
)
similarity = similarity + similarity.T
similarity.setdiag(0)
print(f"  Similarity graph: {similarity.nnz:,} edges")

# Connected components
n_components, labels = connected_components(
    csgraph=similarity, directed=False, return_labels=True
)
print(f"  Found {n_components:,} clusters")

# Build cluster dataframe
cluster_df = pd.DataFrame({
    "cleaned_name": unique_cleaned.values,
    "cluster_id": labels,
})

# Select representative name per cluster (shortest, then alphabetical)
def pick_rep(group):
    sorted_names = sorted(group["cleaned_name"].unique(), key=lambda x: (len(x), x))
    return sorted_names[0] if sorted_names else ""

reps = cluster_df.groupby("cluster_id").apply(pick_rep).reset_index()
reps.columns = ["cluster_id", "representative_name"]
cluster_df = cluster_df.merge(reps, on="cluster_id", how="left")

# Merge cluster info back to cleaning map
full_map = cleaning_map.merge(cluster_df, on="cleaned_name", how="left")

elapsed = time.time() - t0
print(f"  Clustering completed in {elapsed:.1f}s")
print(f"  Clusters: {full_map['cluster_id'].nunique():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Build Mapping Table & Show Sample Clusters

# COMMAND ----------

print(f"\n[5/7] Building mapping table...")
mapping = full_map[["original_name", "cleaned_name", "cluster_id", "representative_name"]].copy()
mapping = mapping.drop_duplicates(subset=["original_name"])

print(f"  Mapping entries:  {len(mapping):,}")
print(f"  Clusters:         {mapping['cluster_id'].nunique():,}")
print(f"  Representatives:  {mapping['representative_name'].nunique():,}")

# Show sample clusters
print(f"\n  Top 5 largest clusters:")
top_clusters = mapping["cluster_id"].value_counts().head(5).index
for cid in top_clusters:
    members = mapping[mapping["cluster_id"] == cid]
    rep = members["representative_name"].iloc[0]
    sample = members["original_name"].head(5).tolist()
    print(f"    Cluster {cid} -> \"{rep}\" ({len(members)} members)")
    for orig in sample:
        print(f"      - \"{orig}\"")

# Display as a Databricks table for interactive exploration
display(spark.createDataFrame(mapping))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Merge Back to Full Dataset (Spark Broadcast Join)

# COMMAND ----------

print(f"\n[6/7] Merging mapping back to {total_rows:,} rows using Spark broadcast join...")
t0 = time.time()

# Convert mapping to Spark DataFrame
mapping_spark = spark.createDataFrame(
    mapping[["original_name", "cleaned_name", "cluster_id", "representative_name"]]
)

# Broadcast join — the mapping table (~250k rows) is small enough to broadcast
result_spark = df_spark.join(
    F.broadcast(mapping_spark),
    df_spark[PAYEE_COLUMN] == mapping_spark["original_name"],
    "left",
).drop("original_name")

result_count = result_spark.count()
elapsed = time.time() - t0
print(f"  Merged in {elapsed:.1f}s")
print(f"  Result rows: {result_count:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Write Results to Delta Tables

# COMMAND ----------

print(f"\n[7/7] Writing results to Delta tables...")
t0 = time.time()

# Write mapping table
mapping_spark.write.mode("overwrite").saveAsTable(OUTPUT_MAPPING_TABLE)
print(f"  Mapping table: {OUTPUT_MAPPING_TABLE} ({len(mapping):,} rows)")

# Write final resolved dataset
result_spark.write.mode("overwrite").saveAsTable(OUTPUT_FINAL_TABLE)
print(f"  Final dataset: {OUTPUT_FINAL_TABLE} ({result_count:,} rows)")

elapsed = time.time() - t0
print(f"  Written in {elapsed:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Summary

# COMMAND ----------

total_time = time.time() - pipeline_start
print(f"\n{'=' * 60}")
print(f"  Pipeline completed in {total_time:.1f}s")
print(f"  Input:      {INPUT_TABLE} ({total_rows:,} rows)")
print(f"  Mapping:    {OUTPUT_MAPPING_TABLE} ({len(mapping):,} entries)")
print(f"  Output:     {OUTPUT_FINAL_TABLE} ({result_count:,} rows)")
print(f"  Clusters:   {mapping['cluster_id'].nunique():,}")
print(f"{'=' * 60}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick Validation: Explore Clusters
# MAGIC
# MAGIC Run the cell below to interactively browse the mapping table.

# COMMAND ----------

display(spark.table(OUTPUT_MAPPING_TABLE).orderBy("cluster_id"))
