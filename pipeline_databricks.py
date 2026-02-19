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

# Clustering parameters (tuned for HIGH PRECISION — fewer false merges)
SIMILARITY_THRESHOLD = 0.85    # cosine similarity cutoff (higher = stricter)
NGRAM_RANGE = (3, 5)           # character n-gram window (narrower = more specific)
TOP_N_PER_ROW = 5              # max neighbours kept per name (fewer = less transitive chaining)
RAPIDFUZZ_THRESHOLD = 75       # RapidFuzz token_sort_ratio cutoff for verification pass

# Pass 3: Re-cluster representatives for RECALL (relaxed thresholds)
RECALL_COSINE_THRESHOLD = 0.60   # looser cosine to catch typo variants
RECALL_NGRAM_RANGE = (2, 4)      # wider n-grams to catch more overlap
RECALL_RAPIDFUZZ_THRESHOLD = 65  # RapidFuzz cutoff for recall pass
# Pass 4: Final Fuzzy Merge (catching spacing/spelling distinct representatives)
RECALL_TOP_N = 10                # more neighbours for broader matching
PASS4_RAPIDFUZZ_THRESHOLD = 60   # Final check for subtle spelling/spacing differences

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
from rapidfuzz import fuzz

# Re-import config after library restart
INPUT_TABLE = "default.payee_raw"
PAYEE_COLUMN = "PayeeName"
OUTPUT_MAPPING_TABLE = "default.payee_mapping"
OUTPUT_FINAL_TABLE   = "default.payee_resolved"
SIMILARITY_THRESHOLD = 0.85
NGRAM_RANGE = (3, 5)
TOP_N_PER_ROW = 5
RAPIDFUZZ_THRESHOLD = 75
RECALL_COSINE_THRESHOLD = 0.60
RECALL_NGRAM_RANGE = (2, 4)
RECALL_RAPIDFUZZ_THRESHOLD = 65
RECALL_TOP_N = 10
PASS4_RAPIDFUZZ_THRESHOLD = 60
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

print(f"\n[4/7] Clustering cleaned names (hierarchical: precision -> recall)...")
t0 = time.time()

# Get unique cleaned names
unique_cleaned = cleaning_map["cleaned_name"].unique()
unique_cleaned = pd.Series([n for n in unique_cleaned if n.strip()], name="cleaned_name")
n = len(unique_cleaned)

print(f"  {n:,} unique cleaned names to cluster")

# ═══════════════════════════════════════════════════════════════════════════
# PASS 1: High-precision cosine similarity (tight threshold)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n  --- Pass 1: Precision clustering (cosine >= {SIMILARITY_THRESHOLD}) ---")

vectorizer_p1 = TfidfVectorizer(
    analyzer="char_wb", ngram_range=NGRAM_RANGE,
    min_df=1, max_df=0.95, sublinear_tf=True, dtype=np.float32,
)
tfidf_p1 = vectorizer_p1.fit_transform(unique_cleaned)
print(f"  TF-IDF: {tfidf_p1.shape} (nnz: {tfidf_p1.nnz:,})")

sim_p1 = awesome_cossim_topn(
    tfidf_p1, tfidf_p1.T, ntop=TOP_N_PER_ROW,
    lower_bound=SIMILARITY_THRESHOLD, use_threads=True, n_jobs=4,
)
sim_p1 = sim_p1 + sim_p1.T
sim_p1.setdiag(0)

n_clusters_p1, labels_p1 = connected_components(csgraph=sim_p1, directed=False, return_labels=True)
print(f"  Pass 1 result: {n_clusters_p1:,} clusters")

# ═══════════════════════════════════════════════════════════════════════════
# PASS 2: RapidFuzz verification (split false merges)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n  --- Pass 2: RapidFuzz verification (split if score < {RAPIDFUZZ_THRESHOLD}) ---")

cluster_df = pd.DataFrame({"cleaned_name": unique_cleaned.values, "cluster_id": labels_p1})
next_cid = cluster_df["cluster_id"].max() + 1
split_count = 0

for cid in cluster_df["cluster_id"].unique():
    members = cluster_df[cluster_df["cluster_id"] == cid]["cleaned_name"].tolist()
    if len(members) <= 1:
        continue
    anchor = min(members, key=lambda x: (len(x), x))
    for member in members:
        if member == anchor:
            continue
        if fuzz.token_sort_ratio(anchor, member) < RAPIDFUZZ_THRESHOLD:
            cluster_df.loc[cluster_df["cleaned_name"] == member, "cluster_id"] = next_cid
            next_cid += 1
            split_count += 1

n_clusters_p2 = cluster_df["cluster_id"].nunique()
print(f"  Split {split_count:,} weak members -> {n_clusters_p2:,} clusters")

# Pick representative per cluster after Pass 2
def pick_rep(group):
    sorted_names = sorted(group["cleaned_name"].unique(), key=lambda x: (len(x), x))
    return sorted_names[0] if sorted_names else ""

reps_p2 = cluster_df.groupby("cluster_id").apply(pick_rep).reset_index()
reps_p2.columns = ["cluster_id", "representative_name"]
cluster_df = cluster_df.merge(reps_p2, on="cluster_id", how="left")

# ═══════════════════════════════════════════════════════════════════════════
# PASS 3: Re-cluster REPRESENTATIVES for RECALL (relaxed thresholds)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n  --- Pass 3: Recall re-clustering (cosine >= {RECALL_COSINE_THRESHOLD}, "
      f"RapidFuzz >= {RECALL_RAPIDFUZZ_THRESHOLD}) ---")

# Get distinct representatives from Pass 2
rep_names = cluster_df[["cluster_id", "representative_name"]].drop_duplicates("cluster_id")
rep_series = rep_names["representative_name"].reset_index(drop=True)
n_reps = len(rep_series)
print(f"  Re-clustering {n_reps:,} representative names...")

# TF-IDF on representatives with wider n-grams
vectorizer_p3 = TfidfVectorizer(
    analyzer="char_wb", ngram_range=RECALL_NGRAM_RANGE,
    min_df=1, max_df=0.95, sublinear_tf=True, dtype=np.float32,
)
tfidf_p3 = vectorizer_p3.fit_transform(rep_series)

sim_p3 = awesome_cossim_topn(
    tfidf_p3, tfidf_p3.T, ntop=RECALL_TOP_N,
    lower_bound=RECALL_COSINE_THRESHOLD, use_threads=True, n_jobs=4,
)
sim_p3 = sim_p3 + sim_p3.T
sim_p3.setdiag(0)

n_super_clusters, super_labels = connected_components(csgraph=sim_p3, directed=False, return_labels=True)
print(f"  Cosine pass: {n_reps:,} reps -> {n_super_clusters:,} super-clusters")

# RapidFuzz verification on super-clusters
rep_cluster_df = pd.DataFrame({
    "representative_name": rep_series.values,
    "super_cluster_id": super_labels,
    "original_cluster_id": rep_names["cluster_id"].values,
})

next_super = rep_cluster_df["super_cluster_id"].max() + 1
recall_split = 0

for scid in rep_cluster_df["super_cluster_id"].unique():
    sc_members = rep_cluster_df[rep_cluster_df["super_cluster_id"] == scid]["representative_name"].tolist()
    if len(sc_members) <= 1:
        continue
    anchor = min(sc_members, key=lambda x: (len(x), x))
    for member in sc_members:
        if member == anchor:
            continue
        if fuzz.token_sort_ratio(anchor, member) < RECALL_RAPIDFUZZ_THRESHOLD:
            rep_cluster_df.loc[rep_cluster_df["representative_name"] == member, "super_cluster_id"] = next_super
            next_super += 1
            recall_split += 1

n_final_super = rep_cluster_df["super_cluster_id"].nunique()
print(f"  RapidFuzz verification split {recall_split:,} -> {n_final_super:,} super-clusters")

# Pick final representative per super-cluster
def pick_super_rep(group):
    sorted_names = sorted(group["representative_name"].unique(), key=lambda x: (len(x), x))
    return sorted_names[0] if sorted_names else ""

super_reps = rep_cluster_df.groupby("super_cluster_id").apply(pick_super_rep).reset_index()
super_reps.columns = ["super_cluster_id", "final_representative"]
rep_cluster_df = rep_cluster_df.merge(super_reps, on="super_cluster_id", how="left")

# Map super-cluster IDs back to the main cluster_df
# old cluster_id -> super_cluster_id + final_representative
merge_map = rep_cluster_df[["original_cluster_id", "super_cluster_id", "final_representative"]].copy()
merge_map.columns = ["cluster_id", "super_cluster_id", "final_representative"]

cluster_df = cluster_df.drop(columns=["representative_name"]).merge(merge_map, on="cluster_id", how="left")
cluster_df = cluster_df.rename(columns={
    "super_cluster_id": "cluster_id_final",
    "final_representative": "representative_name",
})

# ═══════════════════════════════════════════════════════════════════════════
# PASS 4: Final Fuzzy Merge (catching spacing/spelling distinct representatives)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n  --- Pass 4: Final Fuzzy Merge (Cosine ~{RECALL_COSINE_THRESHOLD} + RapidFuzz >= {PASS4_RAPIDFUZZ_THRESHOLD}) ---")

# Step 1: Distinct representatives from Pass 3
p4_reps_df = cluster_df[["cluster_id_final", "representative_name"]].drop_duplicates("cluster_id_final")
p4_rep_series = p4_reps_df["representative_name"].reset_index(drop=True)
n_p4_reps = len(p4_reps_df)
print(f"  Pass 4 input: {n_p4_reps:,} representatives from Pass 3")

# Step 2: TF-IDF + Cosine (reuse Recall Params but fit on new reps)
vectorizer_p4 = TfidfVectorizer(
    analyzer="char_wb", ngram_range=RECALL_NGRAM_RANGE,
    min_df=1, max_df=0.95, sublinear_tf=True, dtype=np.float32,
)
tfidf_p4 = vectorizer_p4.fit_transform(p4_rep_series)

sim_p4 = awesome_cossim_topn(
    tfidf_p4, tfidf_p4.T, ntop=RECALL_TOP_N,
    lower_bound=RECALL_COSINE_THRESHOLD, use_threads=True, n_jobs=4,
)
sim_p4 = sim_p4 + sim_p4.T
sim_p4.setdiag(0)

# Step 3: Connected Components -> Meta Clusters
n_meta_clusters, meta_labels = connected_components(csgraph=sim_p4, directed=False, return_labels=True)
print(f"  Pass 4 Cosine: {n_p4_reps:,} reps -> {n_meta_clusters:,} meta-clusters")

# Step 4: RapidFuzz Verify Meta Clusters
meta_cluster_df = pd.DataFrame({
    "representative_name": p4_rep_series.values,
    "meta_cluster_id": meta_labels,
    "p3_cluster_id": p4_reps_df["cluster_id_final"].values,
})

next_meta = meta_cluster_df["meta_cluster_id"].max() + 1
p4_split = 0

for mcid in meta_cluster_df["meta_cluster_id"].unique():
    mc_members = meta_cluster_df[meta_cluster_df["meta_cluster_id"] == mcid]["representative_name"].tolist()
    if len(mc_members) <= 1:
        continue
    
    anchor = min(mc_members, key=lambda x: (len(x), x))
    for member in mc_members:
        if member == anchor:
            continue
        if fuzz.token_sort_ratio(anchor, member) < PASS4_RAPIDFUZZ_THRESHOLD:
            meta_cluster_df.loc[meta_cluster_df["representative_name"] == member, "meta_cluster_id"] = next_meta
            next_meta += 1
            p4_split += 1

n_final_meta = meta_cluster_df["meta_cluster_id"].nunique()
print(f"  Pass 4 RapidFuzz split {p4_split:,} -> {n_final_meta:,} meta-clusters")

# Step 5: Final Representative Selection
def pick_meta_rep(group):
    sorted_names = sorted(group["representative_name"].unique(), key=lambda x: (len(x), x))
    return sorted_names[0] if sorted_names else ""

meta_reps = meta_cluster_df.groupby("meta_cluster_id").apply(pick_meta_rep).reset_index()
meta_reps.columns = ["meta_cluster_id", "final_representative_p4"]
meta_cluster_df = meta_cluster_df.merge(meta_reps, on="meta_cluster_id", how="left")

# Step 6: Map back to cluster_df
merge_map_p4 = meta_cluster_df[["p3_cluster_id", "meta_cluster_id", "final_representative_p4"]].copy()
merge_map_p4.columns = ["cluster_id_final", "cluster_id_final_p4", "representative_name_p4"]

cluster_df = cluster_df.drop(columns=["representative_name"]).merge(merge_map_p4, on="cluster_id_final", how="left")
cluster_df = cluster_df.drop(columns=["cluster_id_final"]).rename(columns={
    "cluster_id_final_p4": "cluster_id_final",
    "representative_name_p4": "representative_name",
})

# Stats
merged_count = n_clusters_p2 - n_final_super
merged_count_p4 = n_final_super - n_final_meta

print(f"\n  === Hierarchical Clustering Summary ===")
print(f"  Pass 1 (precision cosine):  {n:,} names -> {n_clusters_p1:,} clusters")
print(f"  Pass 2 (RapidFuzz verify):  {n_clusters_p1:,} -> {n_clusters_p2:,} clusters (+{split_count} splits)")
print(f"  Pass 3 (recall re-cluster): {n_clusters_p2:,} -> {n_final_super:,} clusters ({merged_count:,} merged for recall)")
print(f"  Pass 4 (fuzzy merge):       {n_final_super:,} -> {n_final_meta:,} clusters ({merged_count_p4:,} merged for recall)")

# Prepare final output columns
cluster_df = cluster_df[["cleaned_name", "cluster_id_final", "representative_name"]].copy()
cluster_df = cluster_df.rename(columns={"cluster_id_final": "cluster_id"})

# Merge cluster info back to cleaning map
full_map = cleaning_map.merge(cluster_df, on="cleaned_name", how="left")

elapsed = time.time() - t0
print(f"  Total clustering time: {elapsed:.1f}s")
print(f"  Final clusters: {full_map['cluster_id'].nunique():,}")

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
