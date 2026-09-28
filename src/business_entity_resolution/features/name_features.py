"""Deterministic name-level pairwise feature extraction."""

from __future__ import annotations

from typing import Dict, Optional

from business_entity_resolution.features.feature_config import FeatureConfig
from business_entity_resolution.features.similarity_metrics import (
    char_ngram_jaccard,
    difflib_similarity,
    jaro_winkler_similarity,
    levenshtein_similarity,
    token_containment,
    token_jaccard,
    token_set_ratio,
)
from business_entity_resolution.text_normalizer import NormalizedBusinessName


def extract_name_features(
    name1: NormalizedBusinessName,
    name2: NormalizedBusinessName,
    config: Optional[FeatureConfig] = None,
) -> Dict[str, float]:
    """Extract deterministic pairwise features comparing two business names.

    Parameters
    ----------
    name1 : NormalizedBusinessName
        Normalized name of the Source 1 reference entity.
    name2 : NormalizedBusinessName
        Normalized name of the candidate entity (Source 2 or Source 3).
    config : Optional[FeatureConfig]
        Feature extraction configuration.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping feature name to numeric value.
    """
    cfg = config or FeatureConfig()
    features: Dict[str, float] = {}

    is_empty1 = name1.is_empty
    is_empty2 = name2.is_empty

    # If either name is empty, provide deterministic safe defaults
    if is_empty1 or is_empty2:
        return {
            "name_exact_match": 0.0,
            "name_normalized_exact_match": 0.0,
            "name_core_exact_match": 0.0,
            "name_compact_exact_match": 0.0,
            "name_levenshtein_sim": 0.0,
            "name_jaro_winkler_sim": 0.0,
            "name_char_ngram_jaccard": 0.0,
            "name_difflib_sim": 0.0,
            "name_length_ratio": 0.0,
            "name_token_count_diff": float(abs(len(name1.tokens) - len(name2.tokens))),
            "name_token_overlap_count": 0.0,
            "name_token_jaccard": 0.0,
            "name_token_containment": 0.0,
            "name_token_set_sim": 0.0,
            "name_first_token_match": 0.0,
            "name_last_token_match": 0.0,
            "name_consonant_levenshtein_sim": 0.0,
        }

    raw1, raw2 = name1.raw, name2.raw
    norm1, norm2 = name1.normalized, name2.normalized
    core1, core2 = name1.core_name, name2.core_name
    comp1, comp2 = name1.compact, name2.compact
    toks1, toks2 = name1.tokens, name2.tokens

    # 1. Exact & Normalized Equalities
    features["name_exact_match"] = 1.0 if raw1 == raw2 else 0.0
    features["name_normalized_exact_match"] = 1.0 if norm1 == norm2 else 0.0
    features["name_core_exact_match"] = 1.0 if core1 == core2 else 0.0
    features["name_compact_exact_match"] = 1.0 if comp1 == comp2 else 0.0

    # 2. String Similarities
    features["name_levenshtein_sim"] = round(levenshtein_similarity(norm1, norm2), cfg.float_precision)
    features["name_jaro_winkler_sim"] = round(
        jaro_winkler_similarity(
            norm1,
            norm2,
            prefix_weight=cfg.jaro_winkler_prefix_weight,
            max_prefix=cfg.jaro_winkler_max_prefix,
            threshold=cfg.jaro_winkler_threshold,
        ),
        cfg.float_precision,
    )
    features["name_char_ngram_jaccard"] = round(char_ngram_jaccard(comp1, comp2, n=cfg.ngram_size), cfg.float_precision)
    features["name_difflib_sim"] = round(difflib_similarity(norm1, norm2), cfg.float_precision)
    
    def extract_consonants(text: str) -> str:
        vowels = set("aeiouyAEIOUY .-,_")
        return "".join(c for c in text if c not in vowels)
        
    cons1 = extract_consonants(norm1)
    cons2 = extract_consonants(norm2)
    features["name_consonant_levenshtein_sim"] = round(levenshtein_similarity(cons1, cons2), cfg.float_precision)

    len1, len2 = len(norm1), len(norm2)
    features["name_length_ratio"] = round(min(len1, len2) / max(len1, len2), cfg.float_precision) if max(len1, len2) > 0 else 0.0

    # 3. Token-Level Features
    features["name_token_count_diff"] = float(abs(len(toks1) - len(toks2)))
    set1, set2 = set(toks1), set(toks2)
    overlap = len(set1.intersection(set2))
    features["name_token_overlap_count"] = float(overlap)
    features["name_token_jaccard"] = round(token_jaccard(toks1, toks2), cfg.float_precision)
    features["name_token_containment"] = round(token_containment(toks1, toks2), cfg.float_precision)
    features["name_token_set_sim"] = round(token_set_ratio(toks1, toks2), cfg.float_precision)

    features["name_first_token_match"] = 1.0 if (toks1 and toks2 and toks1[0] == toks2[0]) else 0.0
    features["name_last_token_match"] = 1.0 if (toks1 and toks2 and toks1[-1] == toks2[-1]) else 0.0

    return features
