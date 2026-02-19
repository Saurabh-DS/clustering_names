"""
Central configuration for the PayeeName Entity Resolution Pipeline.
"""
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

SYNTHETIC_DATA_PATH = DATA_DIR / "synthetic_payees.csv"
MAPPING_TABLE_PATH = OUTPUT_DIR / "mapping_table.csv"
FINAL_DATASET_PATH = OUTPUT_DIR / "final_dataset.csv"

# ─── Synthetic Data Settings ──────────────────────────────────────────────────
NUM_UNIQUE_NAMES = 250_000       # target unique dirty variants
NUM_TOTAL_ROWS = 1_000_000       # total rows (set to 30_000_000 for production)
CANONICAL_NAMES_COUNT = 200      # number of "ground truth" base names

# ─── Cleaning Settings ───────────────────────────────────────────────────────
PREFIXES_TO_REMOVE = [
    "mr", "mrs", "miss", "ms", "dr", "moham",
]

JUNK_WORDS_TO_REMOVE = [
    "bacs", "micl", "postcodeservices", "ltd", "limited",
]

# ─── Clustering Settings ─────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.85      # cosine similarity cutoff (higher = stricter)
NGRAM_RANGE = (3, 5)             # character n-gram window (narrower = more specific)
TOP_N_PER_ROW = 5                # max neighbours kept per name (fewer = less transitive chaining)
MIN_CLUSTER_SIZE = 1             # clusters smaller than this are singletons
RAPIDFUZZ_THRESHOLD = 75         # token_sort_ratio cutoff for verification pass

# Pass 3: Re-cluster representatives for RECALL (relaxed thresholds)
RECALL_COSINE_THRESHOLD = 0.60
RECALL_NGRAM_RANGE = (2, 4)
RECALL_RAPIDFUZZ_THRESHOLD = 65
RECALL_TOP_N = 10

# Pass 4: Final Fuzzy Merge (catching spacing/spelling distinct representatives)
PASS4_RAPIDFUZZ_THRESHOLD = 60


# ─── Performance ──────────────────────────────────────────────────────────────
USE_POLARS_FOR_MERGE = True      # use polars for the 30M-row merge step
CHUNK_SIZE = 500_000             # rows per chunk when reading large CSVs
