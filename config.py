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

# ─── Clustering Settings ─────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.70      # cosine similarity cutoff for edges
NGRAM_RANGE = (2, 4)             # character n-gram window
TOP_N_PER_ROW = 10               # max neighbours kept per name in sparse sim
MIN_CLUSTER_SIZE = 1             # clusters smaller than this are singletons

# ─── Performance ──────────────────────────────────────────────────────────────
USE_POLARS_FOR_MERGE = True      # use polars for the 30M-row merge step
CHUNK_SIZE = 500_000             # rows per chunk when reading large CSVs
