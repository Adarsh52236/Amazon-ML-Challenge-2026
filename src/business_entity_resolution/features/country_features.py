"""Deterministic country compatibility pairwise feature extraction."""

from __future__ import annotations

from typing import Dict, Optional

from business_entity_resolution.features.feature_config import FeatureConfig


def extract_country_features(
    country1: Optional[str],
    country2: Optional[str],
    config: Optional[FeatureConfig] = None,
) -> Dict[str, float]:
    """Extract deterministic pairwise features comparing two country values.

    Parameters
    ----------
    country1 : Optional[str]
        Raw or normalized country of the Source 1 reference entity.
    country2 : Optional[str]
        Raw or normalized country of the candidate entity.
    config : Optional[FeatureConfig]
        Feature extraction configuration.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping feature name to numeric value.
    """
    c1 = "" if country1 is None else str(country1).strip().lower()
    c2 = "" if country2 is None else str(country2).strip().lower()

    is_missing1 = c1 == "" or c1 in {"nan", "null", "none"}
    is_missing2 = c2 == "" or c2 in {"nan", "null", "none"}

    features: Dict[str, float] = {
        "country_missing_source": 1.0 if is_missing1 else 0.0,
        "country_missing_target": 1.0 if is_missing2 else 0.0,
        "country_both_missing": 1.0 if (is_missing1 and is_missing2) else 0.0,
    }

    if is_missing1 or is_missing2:
        features["country_exact_match"] = 0.0
        features["country_conflict"] = 0.0
    else:
        is_match = (c1 == c2)
        features["country_exact_match"] = 1.0 if is_match else 0.0
        features["country_conflict"] = 0.0 if is_match else 1.0

    return features
