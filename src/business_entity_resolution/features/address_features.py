"""Deterministic address-level pairwise feature extraction."""

from __future__ import annotations

from typing import Dict, Optional

from business_entity_resolution.address_normalizer import NormalizedBusinessAddress
from business_entity_resolution.features.feature_config import FeatureConfig
from business_entity_resolution.features.similarity_metrics import (
    levenshtein_similarity,
    token_jaccard,
)


def extract_address_features(
    addr1: NormalizedBusinessAddress,
    addr2: NormalizedBusinessAddress,
    country: Optional[str] = None,
    config: Optional[FeatureConfig] = None,
) -> Dict[str, float]:
    """Extract deterministic pairwise features comparing two business addresses.

    Parameters
    ----------
    addr1 : NormalizedBusinessAddress
        Normalized address of the Source 1 reference entity.
    addr2 : NormalizedBusinessAddress
        Normalized address of the candidate entity (Source 2 or Source 3).
    country : Optional[str]
        Country context to calibrate postal prefix comparisons.
    config : Optional[FeatureConfig]
        Feature extraction configuration.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping feature name to numeric value.
    """
    cfg = config or FeatureConfig()
    features: Dict[str, float] = {}

    is_empty1 = addr1.is_empty
    is_empty2 = addr2.is_empty

    # Data Quality Indicators (always computed regardless of missingness)
    features["address_missing_source"] = 1.0 if is_empty1 else 0.0
    features["address_missing_target"] = 1.0 if is_empty2 else 0.0
    features["address_both_missing"] = 1.0 if (is_empty1 and is_empty2) else 0.0
    features["address_both_present"] = 1.0 if (not is_empty1 and not is_empty2) else 0.0

    toks1, toks2 = addr1.tokens, addr2.tokens
    norm1, norm2 = addr1.normalized, addr2.normalized
    comp1 = norm1.replace(" ", "")
    comp2 = norm2.replace(" ", "")

    features["address_token_count_diff"] = float(abs(len(toks1) - len(toks2)))
    
    import re
    def extract_street_num(raw: str):
        if not raw: return ""
        # match first token starting with a digit
        for t in raw.replace(",", " ").split():
            if t and t[0].isdigit():
                return t.strip(".,;:").upper()
        return ""
        
    num_ext1 = extract_street_num(addr1.raw)
    num_ext2 = extract_street_num(addr2.raw)
    
    def extract_numeric_comp(num_str: str):
        if not num_str: return ""
        return "".join(re.findall(r"\d+", num_str))
        
    num_comp1 = extract_numeric_comp(num_ext1)
    num_comp2 = extract_numeric_comp(num_ext2)
    
    def get_num_tokens(raw: str):
        if not raw: return set()
        return set(re.findall(r"\d+", raw))
        
    ntoks1 = get_num_tokens(addr1.raw)
    ntoks2 = get_num_tokens(addr2.raw)

    features["address_length_diff"] = float(abs(len(norm1) - len(norm2)))

    # If either address is missing, comparisons are 0.0 (safe defaults)
    if is_empty1 or is_empty2:
        features["address_normalized_exact_match"] = 0.0
        features["address_compact_exact_match"] = 0.0
        features["address_street_number_match"] = 0.0
        features["address_street_number_both_present"] = 0.0
        features["address_street_number_mismatch"] = 0.0
        features["address_postal_exact_match"] = 0.0
        features["address_postal_prefix_match"] = 0.0
        features["address_postal_both_present"] = 0.0
        features["address_postal_mismatch"] = 0.0
        features["address_street_stem_match"] = 0.0
        features["address_token_overlap_count"] = 0.0
        features["address_token_jaccard"] = 0.0
        features["address_levenshtein_sim"] = 0.0
        
        # New features defaults for missing
        features["address_exact_street_number_match"] = -1.0
        features["address_street_number_numeric_match"] = -1.0
        features["address_postal_code_exact_match"] = -1.0
        features["address_postal_code_prefix_match"] = -1.0
        features["address_numeric_token_jaccard"] = -1.0
        features["address_number_consistency"] = -1.0
        return features

    # 1. Exact & Normalized Equalities
    features["address_normalized_exact_match"] = 1.0 if norm1 == norm2 else 0.0
    features["address_compact_exact_match"] = 1.0 if comp1 == comp2 else 0.0

    # 2. Structured Component Comparisons
    num1, num2 = addr1.street_number, addr2.street_number
    both_have_num = bool(num1 and num2)
    features["address_street_number_both_present"] = 1.0 if both_have_num else 0.0
    features["address_street_number_match"] = 1.0 if (both_have_num and num1 == num2) else 0.0
    features["address_street_number_mismatch"] = 1.0 if (both_have_num and num1 != num2) else 0.0

    p1, p2 = addr1.postal_code, addr2.postal_code
    both_have_postal = bool(p1 and p2)
    features["address_postal_both_present"] = 1.0 if both_have_postal else 0.0
    features["address_postal_exact_match"] = 1.0 if (both_have_postal and p1 == p2) else 0.0
    features["address_postal_mismatch"] = 1.0 if (both_have_postal and p1 != p2) else 0.0

    # Postal Prefix (first 2 digits for US/France, first 3 for India)
    prefix_len = 3 if country and "india" in country.lower() else 2
    prefix1 = p1[:prefix_len] if len(p1) >= prefix_len else p1
    prefix2 = p2[:prefix_len] if len(p2) >= prefix_len else p2
    features["address_postal_prefix_match"] = (
        1.0 if (both_have_postal and prefix1 and prefix1 == prefix2) else 0.0
    )

    stem1, stem2 = addr1.street_stem, addr2.street_stem
    both_have_stem = bool(stem1 and stem2)
    features["address_street_stem_match"] = 1.0 if (both_have_stem and stem1 == stem2) else 0.0

    # 3. Token Overlap & Similarity
    set1, set2 = set(toks1), set(toks2)
    features["address_token_overlap_count"] = float(len(set1.intersection(set2)))
    features["address_token_jaccard"] = round(token_jaccard(toks1, toks2), cfg.float_precision)
    features["address_levenshtein_sim"] = round(levenshtein_similarity(norm1, norm2), cfg.float_precision)

    return features
