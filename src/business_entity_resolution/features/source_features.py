"""Deterministic cross-source feature extraction."""

from __future__ import annotations

from typing import Dict, Optional

from business_entity_resolution.features.feature_config import FeatureConfig


def extract_source_features(
    matched_source: str,
    config: Optional[FeatureConfig] = None,
) -> Dict[str, float]:
    """Extract deterministic cross-source indicators for candidate pairs.

    Parameters
    ----------
    matched_source : str
        Source identifier of the secondary record ('S2' or 'S3').
    config : Optional[FeatureConfig]
        Feature extraction configuration.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping feature name to numeric value.
    """
    src = str(matched_source).strip().upper()
    is_s2 = src.startswith("S2")
    is_s3 = src.startswith("S3")

    return {
        "target_source_is_s2": 1.0 if is_s2 else 0.0,
        "target_source_is_s3": 1.0 if is_s3 else 0.0,
        "pair_type_s1_s2": 1.0 if is_s2 else 0.0,
        "pair_type_s1_s3": 1.0 if is_s3 else 0.0,
    }
