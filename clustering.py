"""
Clustering Module for PayeeName Entity Resolution
===================================================
Uses TF-IDF character n-grams + sparse cosine similarity to group
similar cleaned names into clusters — without the 62-billion-comparison
nested loop.

Pipeline:
  1. TF-IDF vectorise cleaned names (char_wb n-grams)
  2. Compute sparse cosine similarity via sparse_dot_topn
  3. Build adjacency graph from similarity matrix
  4. Find connected components → cluster IDs
  5. Select representative name per cluster (most frequent)
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

import config


# ─── TF-IDF Vectorization ────────────────────────────────────────────────────

def build_tfidf_matrix(
    names: pd.Series,
    ngram_range: tuple = None,
) -> csr_matrix:
    """
    Build a TF-IDF matrix from a Series of cleaned name strings
    using character-level n-grams with word boundaries.

    Parameters
    ----------
    names : pd.Series
        Cleaned name strings (lowercase, no noise).
    ngram_range : tuple
        Min and max n-gram sizes, default from config.

    Returns
    -------
    csr_matrix
        Sparse TF-IDF matrix of shape (n_names, n_features).
    TfidfVectorizer
        The fitted vectorizer (for inspection if needed).
    """
    ngram_range = ngram_range or config.NGRAM_RANGE

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=1,
        max_df=0.95,       # ignore n-grams in >95% of names
        sublinear_tf=True,  # apply log normalization
        dtype=np.float32,   # save memory
    )

    print(f"  Fitting TF-IDF vectorizer (n-gram range: {ngram_range})...")
    tfidf_matrix = vectorizer.fit_transform(names)
    print(f"  Matrix shape: {tfidf_matrix.shape}  "
          f"(nnz: {tfidf_matrix.nnz:,}, density: {tfidf_matrix.nnz / np.prod(tfidf_matrix.shape):.6f})")

    return tfidf_matrix, vectorizer


# ─── Sparse Cosine Similarity ────────────────────────────────────────────────

def compute_similarity(
    tfidf_matrix: csr_matrix,
    threshold: float = None,
    top_n: int = None,
) -> csr_matrix:
    """
    Compute cosine similarity using sparse_dot_topn.
    Only keeps similarities >= threshold and at most top_n per row.

    Returns a sparse symmetric similarity matrix.
    """
    threshold = threshold or config.SIMILARITY_THRESHOLD
    top_n = top_n or config.TOP_N_PER_ROW

    n = tfidf_matrix.shape[0]
    print(f"  Computing sparse cosine similarity "
          f"(threshold={threshold}, top_n={top_n})...")
    print(f"  This is {n:,} names — using matrix multiplication, not nested loops.")

    # awesome_cossim_topn computes A * B^T but keeps only top-N per row above threshold
    similarity = awesome_cossim_topn(
        tfidf_matrix,
        tfidf_matrix.T,
        ntop=top_n,
        lower_bound=threshold,
        use_threads=True,
        n_jobs=4,
    )

    # Make it symmetric (the function may not return a perfectly symmetric matrix)
    similarity = similarity + similarity.T
    similarity.setdiag(0)  # remove self-similarities

    print(f"  Similarity graph: {similarity.nnz:,} edges")
    return similarity


# ─── Connected Components Clustering ─────────────────────────────────────────

def find_clusters(similarity_matrix: csr_matrix) -> np.ndarray:
    """
    Find connected components in the similarity graph.
    Each component = one cluster of similar names.

    Returns an array of cluster IDs (one per name).
    """
    print(f"  Finding connected components...")
    n_components, labels = connected_components(
        csgraph=similarity_matrix,
        directed=False,
        return_labels=True,
    )
    print(f"  Found {n_components:,} clusters from {len(labels):,} names")

    # Summary stats
    unique, counts = np.unique(labels, return_counts=True)
    multi_member = counts[counts > 1]
    print(f"  Singletons: {np.sum(counts == 1):,}")
    print(f"  Multi-member clusters: {len(multi_member):,} "
          f"(avg size: {multi_member.mean():.1f}, max: {multi_member.max()})")

    return labels


# ─── Representative Name Selection ───────────────────────────────────────────

def select_representatives(
    df: pd.DataFrame,
    name_col: str = "cleaned_name",
    cluster_col: str = "cluster_id",
) -> pd.DataFrame:
    """
    For each cluster, select the representative name.
    Strategy: pick the shortest name (ties broken by alphabetical order).
    The shortest cleaned name is usually the "purest" version.

    Adds a `representative_name` column to the dataframe.
    """
    print(f"  Selecting representative names per cluster...")

    def _pick_representative(group):
        # Sort by length (ascending), then alphabetically
        sorted_names = sorted(group[name_col].unique(), key=lambda x: (len(x), x))
        return sorted_names[0] if sorted_names else ""

    reps = (
        df.groupby(cluster_col)
        .apply(_pick_representative)
        .reset_index()
    )
    reps.columns = [cluster_col, "representative_name"]

    df = df.merge(reps, on=cluster_col, how="left")
    return df


# ─── Main Clustering Pipeline ────────────────────────────────────────────────

def cluster_names(
    cleaned_names: pd.Series,
    threshold: float = None,
    ngram_range: tuple = None,
    top_n: int = None,
) -> pd.DataFrame:
    """
    Full clustering pipeline: vectorize → similarity → components → representatives.

    Parameters
    ----------
    cleaned_names : pd.Series
        Unique, cleaned name strings.
    threshold : float
        Cosine similarity cutoff (default from config).
    ngram_range : tuple
        Character n-gram range (default from config).
    top_n : int
        Max neighbours per name (default from config).

    Returns
    -------
    pd.DataFrame
        Columns: cleaned_name, cluster_id, representative_name
    """
    threshold = threshold or config.SIMILARITY_THRESHOLD
    ngram_range = ngram_range or config.NGRAM_RANGE
    top_n = top_n or config.TOP_N_PER_ROW

    # Drop empty strings and duplicates
    names = cleaned_names.dropna().unique()
    names = pd.Series([n for n in names if n.strip()], name="cleaned_name")
    n = len(names)

    print(f"\n{'='*60}")
    print(f"  Clustering {n:,} unique cleaned names")
    print(f"{'='*60}\n")

    # Step 1: TF-IDF
    tfidf_matrix, vectorizer = build_tfidf_matrix(names, ngram_range)

    # Step 2: Sparse cosine similarity
    similarity = compute_similarity(tfidf_matrix, threshold, top_n)

    # Step 3: Connected components
    labels = find_clusters(similarity)

    # Step 4: Build result dataframe
    result = pd.DataFrame({
        "cleaned_name": names.values,
        "cluster_id": labels,
    })

    # Step 5: Pick representatives
    result = select_representatives(result)

    print(f"\n  Clustering complete.")
    print(f"  Input names: {n:,}")
    print(f"  Clusters:    {result['cluster_id'].nunique():,}")
    print(f"  Representative names: {result['representative_name'].nunique():,}")

    return result


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Small test
    test_names = pd.Series([
        "john smith",
        "john smth",
        "jon smith",
        "john smith",
        "sarah connor",
        "sarah conner",
        "sara connor",
        "aviva insurance",
        "aviva ins",
        "avvia insurance",
        "direct line group",
        "direct line grp",
        "irwin mitchell",
        "irwn mitchell",
        "irwin mitchel",
    ])

    result = cluster_names(test_names, threshold=0.5)
    print("\n" + result.to_string(index=False))
