"""Deterministic string and token similarity metrics for pairwise entity resolution.

Uses rapidfuzz (C++ backend) when available for 5-10x speedup over pure Python.
Falls back to pure-Python implementations if rapidfuzz is not installed.
"""

from __future__ import annotations

import difflib
from typing import Sequence, Set, Tuple

# ─── rapidfuzz fast path (installed via: pip install rapidfuzz) ───────────────
try:
    from rapidfuzz.distance import Levenshtein as _rf_lev
    from rapidfuzz.distance import JaroWinkler as _rf_jw
    from rapidfuzz import fuzz as _rf_fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


# ─── Levenshtein ──────────────────────────────────────────────────────────────

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the minimum edit distance between two strings."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    if _HAS_RAPIDFUZZ:
        return _rf_lev.distance(s1, s2)

    # Pure-Python single-row DP fallback
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    len1, len2 = len(s1), len(s2)
    current_row = list(range(len1 + 1))

    for i in range(1, len2 + 1):
        c2 = s2[i - 1]
        previous_row = current_row
        current_row = [i] + [0] * len1
        for j in range(1, len1 + 1):
            c1 = s1[j - 1]
            cost = 0 if c1 == c2 else 1
            current_row[j] = min(
                previous_row[j] + 1,
                current_row[j - 1] + 1,
                previous_row[j - 1] + cost,
            )

    return current_row[len1]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Compute normalized edit similarity in range [0.0, 1.0].

    sim = 1.0 - (edit_dist / max(len1, len2)).
    Returns 1.0 if both strings are empty, or 0.0 if one is empty.
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    if _HAS_RAPIDFUZZ:
        return _rf_lev.normalized_similarity(s1, s2)

    max_len = max(len(s1), len(s2))
    dist = levenshtein_distance(s1, s2)
    return max(0.0, 1.0 - (dist / max_len))


# ─── Jaro / Jaro-Winkler ─────────────────────────────────────────────────────

def jaro_distance(s1: str, s2: str) -> float:
    """Compute the standard Jaro similarity metric in range [0.0, 1.0]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    if _HAS_RAPIDFUZZ:
        from rapidfuzz.distance import Jaro as _rf_jaro
        return _rf_jaro.normalized_similarity(s1, s2)

    # Pure-Python fallback
    len1, len2 = len(s1), len(s2)
    match_distance = max(len1, len2) // 2 - 1

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    k = 0
    transpositions = 0
    for i in range(len1):
        if s1_matches[i]:
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

    transpositions //= 2

    return (
        (matches / len1) + (matches / len2) + ((matches - transpositions) / matches)
    ) / 3.0


def jaro_winkler_similarity(
    s1: str,
    s2: str,
    prefix_weight: float = 0.1,
    max_prefix: int = 4,
    threshold: float = 0.7,
) -> float:
    """Compute Jaro-Winkler similarity with common prefix bonus."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    if _HAS_RAPIDFUZZ:
        # rapidfuzz uses prefix_weight=0.1 by default; expose parameter
        return _rf_jw.normalized_similarity(s1, s2, prefix_weight=prefix_weight)

    # Pure-Python fallback
    jaro_sim = jaro_distance(s1, s2)
    if jaro_sim < threshold:
        return jaro_sim

    prefix_len = 0
    for c1, c2 in zip(s1[:max_prefix], s2[:max_prefix]):
        if c1 == c2:
            prefix_len += 1
        else:
            break

    return min(1.0, jaro_sim + (prefix_len * prefix_weight * (1.0 - jaro_sim)))


# ─── Character n-gram Jaccard ─────────────────────────────────────────────────

def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Compute character n-gram Jaccard similarity.

    Used on compact business strings (e.g. 'apexdatasystems' vs 'apexdata').
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    if len(s1) < n and len(s2) < n:
        return 1.0 if s1 == s2 else 0.0

    ngrams1 = {s1[i : i + n] for i in range(max(0, len(s1) - n + 1))} if len(s1) >= n else {s1}
    ngrams2 = {s2[i : i + n] for i in range(max(0, len(s2) - n + 1))} if len(s2) >= n else {s2}

    intersection = len(ngrams1.intersection(ngrams2))
    union = len(ngrams1.union(ngrams2))
    return float(intersection / union) if union > 0 else 0.0


# ─── Difflib / Sequence matching ─────────────────────────────────────────────

def difflib_similarity(s1: str, s2: str) -> float:
    """Compute Gestalt pattern matching ratio.

    Uses rapidfuzz.fuzz.ratio (C++ implementation) when available,
    otherwise falls back to Python's difflib.SequenceMatcher.
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    if _HAS_RAPIDFUZZ:
        return _rf_fuzz.ratio(s1, s2) / 100.0

    return difflib.SequenceMatcher(None, s1, s2).ratio()


# ─── Token-level metrics ──────────────────────────────────────────────────────

def token_jaccard(tokens1: Sequence[str], tokens2: Sequence[str]) -> float:
    """Compute Jaccard similarity across token sets."""
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0

    set1 = set(tokens1)
    set2 = set(tokens2)
    if set1 == set2:
        return 1.0

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection / union) if union > 0 else 0.0


def token_containment(tokens1: Sequence[str], tokens2: Sequence[str]) -> float:
    """Compute token containment ratio: |intersection| / min(|set1|, |set2|).

    Returns 1.0 if the shorter entity's tokens are entirely contained in the longer entity.
    """
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0

    set1 = set(tokens1)
    set2 = set(tokens2)
    min_len = min(len(set1), len(set2))
    if min_len == 0:
        return 0.0

    intersection = len(set1.intersection(set2))
    return float(intersection / min_len)


def token_set_ratio(tokens1: Sequence[str], tokens2: Sequence[str]) -> float:
    """Compute token set similarity: compares sorted unique intersection tokens against the rest."""
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0

    set1 = set(tokens1)
    set2 = set(tokens2)
    intersection = set1.intersection(set2)
    if not intersection:
        return 0.0

    sorted_inter = " ".join(sorted(intersection))
    str1 = " ".join(sorted(set1))
    str2 = " ".join(sorted(set2))

    ratio1 = difflib_similarity(sorted_inter, str1)
    ratio2 = difflib_similarity(sorted_inter, str2)
    return max(ratio1, ratio2)
