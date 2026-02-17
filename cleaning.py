"""
Cleaning Module for PayeeName Entity Resolution
=================================================
Provides vectorized string cleaning operations:
  1. Prefix removal (Mr, Mrs, Miss, Ms, Dr, Moham) — whole-word only
  2. Date pattern removal
  3. Noise stripping (leading/trailing punctuation, digits, special chars)
  4. Lowercase + whitespace standardization
"""
import re
import pandas as pd

import config


# ─── Compiled Regex Patterns ─────────────────────────────────────────────────

# Prefix pattern: match whole words only, optional trailing period
_PREFIX_WORDS = "|".join(re.escape(p) for p in config.PREFIXES_TO_REMOVE)
RE_PREFIXES = re.compile(
    rf"\b({_PREFIX_WORDS})\b\.?\s*",
    re.IGNORECASE,
)

# Date patterns — covers most common formats
RE_DATES = re.compile(
    r"""
    (?:                                     # Group of alternatives
        \d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}   # DD/MM/YYYY, MM-DD-YY, etc.
      | \d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}     # YYYY-MM-DD
      | \d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun   # DD Mon YYYY
           |Jul|Aug|Sep|Oct|Nov|Dec)
           [a-z]*\s+\d{2,4}
      | (?:Jan|Feb|Mar|Apr|May|Jun              # Mon DD, YYYY
           |Jul|Aug|Sep|Oct|Nov|Dec)
           [a-z]*\s+\d{1,2},?\s+\d{2,4}
      | \d{8}                                   # DDMMYYYY (8 consecutive digits)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Leading/trailing noise: non-alphanumeric and non-space characters
RE_LEADING_NOISE = re.compile(r"^[^a-zA-Z]+")
RE_TRAILING_NOISE = re.compile(r"[^a-zA-Z]+$")

# Multiple spaces
RE_MULTI_SPACE = re.compile(r"\s{2,}")


# ─── Cleaning Functions ──────────────────────────────────────────────────────

def _remove_prefixes(text: str) -> str:
    """Remove honourific prefixes as whole words."""
    return RE_PREFIXES.sub("", text)


def _remove_dates(text: str) -> str:
    """Remove embedded date patterns."""
    return RE_DATES.sub("", text)


def _remove_noise(text: str) -> str:
    """Strip leading/trailing non-alpha characters."""
    text = RE_LEADING_NOISE.sub("", text)
    text = RE_TRAILING_NOISE.sub("", text)
    return text


def _standardize(text: str) -> str:
    """Lowercase, collapse spaces, strip."""
    text = text.lower()
    text = RE_MULTI_SPACE.sub(" ", text)
    return text.strip()


def clean_single(text: str) -> str:
    """
    Apply the full cleaning pipeline to a single string.

    Order matters:
      1. Remove dates first (they contain digits that would be caught by noise removal)
      2. Remove prefixes
      3. Strip noise characters
      4. Standardize case and whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = _remove_dates(text)
    text = _remove_prefixes(text)
    text = _remove_noise(text)
    text = _standardize(text)
    return text


def clean_names(names: pd.Series) -> pd.Series:
    """
    Vectorized cleaning of a pandas Series of name strings.

    Parameters
    ----------
    names : pd.Series
        Raw payee name strings.

    Returns
    -------
    pd.Series
        Cleaned name strings, same index as input.
    """
    return names.fillna("").astype(str).map(clean_single)


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_names = pd.Series([
        "Mr. John Doe 12/01/2023",
        "DR SARAH CONNOR!!!",
        "##Miss Jane 01-12-99",
        "  moham ahmed hussein  ",
        "Aviva Insurance 15 Jan 2020",
        "!!!123 Irwin Mitchell LLP ---",
        "Mrs. Raj Patel 23/04/2019",
        "DIRECT LINE GROUP",
        "Jon   Doe",
        "mr.john smith",
        "Mistral Technologies",    # should NOT strip "Mis" from "Mistral"
        "Drummond Partners",       # should NOT strip "Dr" from "Drummond"
    ])

    cleaned = clean_names(test_names)

    print("\n{:<45s} | {:<30s}".format("ORIGINAL", "CLEANED"))
    print("-" * 80)
    for orig, clean in zip(test_names, cleaned):
        print(f"{orig:<45s} | {clean:<30s}")
