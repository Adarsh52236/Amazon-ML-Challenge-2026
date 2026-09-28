"""Deterministic candidate blocking provenance feature extraction."""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from business_entity_resolution.features.feature_config import FeatureConfig


def extract_provenance_features(
    channels: Optional[List[str]],
    config: Optional[FeatureConfig] = None,
) -> Dict[str, float]:
    """Extract candidate-generation provenance features from the list of triggered blocking channels.

    Parameters
    ----------
    channels : Optional[List[str]]
        List of blocking channel identifiers that nominated this candidate pair.
    config : Optional[FeatureConfig]
        Feature extraction configuration.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping feature name to numeric value.
    """
    cfg = config or FeatureConfig()
    ch_list = channels or []
    ch_set: Set[str] = set(ch_list)

    weights = cfg.channel_priority_weights
    max_weight = max((weights.get(ch, 1) for ch in ch_set), default=0)

    features: Dict[str, float] = {
        "prov_channel_count": float(len(ch_set)),
        "prov_max_channel_weight": float(max_weight),
        "prov_has_exact_name": 1.0 if "exact_name" in ch_set else 0.0,
        "prov_has_country_exact_name": 1.0 if "country_exact_name" in ch_set else 0.0,
        "prov_has_country_core_name": 1.0 if "country_core_name" in ch_set else 0.0,
        "prov_has_country_compact": 1.0 if "country_compact" in ch_set else 0.0,
        "prov_has_token_shingles": 1.0 if "name_token_shingles" in ch_set else 0.0,
        "prov_has_addr_num_stem": 1.0 if "addr_num_stem" in ch_set else 0.0,
        "prov_has_postal_num": 1.0 if "postal_num" in ch_set else 0.0,
        "prov_has_postal_prefix": 1.0 if "postal_prefix" in ch_set else 0.0,
    }

    return features
