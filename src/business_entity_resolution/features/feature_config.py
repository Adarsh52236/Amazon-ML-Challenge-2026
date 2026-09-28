"""Centralized configuration for pairwise feature extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set


@dataclass
class FeatureConfig:
    """Configuration governing feature generation, enabling/disabling feature groups,

    precision policy, and missing value strategies.
    """

    # Feature group toggles
    enable_name_features: bool = True
    enable_address_features: bool = True
    enable_country_features: bool = True
    enable_provenance_features: bool = True
    enable_cross_source_features: bool = True

    # String similarity settings
    ngram_size: int = 3
    jaro_winkler_prefix_weight: float = 0.1
    jaro_winkler_max_prefix: int = 4
    jaro_winkler_threshold: float = 0.7

    # Precision & Missing Value Policy
    float_precision: int = 4
    default_missing_numeric: float = 0.0

    # Channel priority weights (aligned with Phase 2.6 blocking configuration)
    channel_priority_weights: dict[str, int] = field(
        default_factory=lambda: {
            "country_exact_name": 10,
            "exact_name": 10,
            "country_core_name": 8,
            "country_compact": 6,
            "addr_num_stem": 6,
            "postal_num": 5,
            "postal_prefix": 4,
            "name_token_shingles": 4,
        }
    )

    # Batch processing parameters
    batch_size: int = 50_000

    def get_feature_names(self) -> List[str]:
        """Return the deterministic, ordered list of all feature names enabled under this config."""
        features: List[str] = []

        if self.enable_name_features:
            features.extend([
                # Exact & normalized equality
                "name_exact_match",
                "name_normalized_exact_match",
                "name_core_exact_match",
                "name_compact_exact_match",
                # String similarity metrics
                "name_levenshtein_sim",
                "name_jaro_winkler_sim",
                "name_char_ngram_jaccard",
                "name_difflib_sim",
                "name_length_ratio",
                # Token-level features
                "name_token_count_diff",
                "name_token_overlap_count",
                "name_token_jaccard",
                "name_token_containment",
                "name_token_set_sim",
                "name_first_token_match",
                "name_last_token_match",
            ])

        if self.enable_address_features:
            features.extend([
                # Exact & normalized equality
                "address_normalized_exact_match",
                "address_compact_exact_match",
                # Structured component matches
                "address_street_number_match",
                "address_street_number_both_present",
                "address_street_number_mismatch",
                "address_postal_exact_match",
                "address_postal_prefix_match",
                "address_postal_both_present",
                "address_postal_mismatch",
                "address_street_stem_match",
                # Similarity & overlap
                "address_token_overlap_count",
                "address_token_jaccard",
                "address_levenshtein_sim",
                # Data quality & missingness indicators
                "address_missing_source",
                "address_missing_target",
                "address_both_missing",
                "address_both_present",
                "address_token_count_diff",
                "address_length_diff",
            ])

        if self.enable_country_features:
            features.extend([
                "country_exact_match",
                "country_missing_source",
                "country_missing_target",
                "country_both_missing",
                "country_conflict",
            ])

        if self.enable_provenance_features:
            features.extend([
                "prov_channel_count",
                "prov_max_channel_weight",
                "prov_has_exact_name",
                "prov_has_country_exact_name",
                "prov_has_country_core_name",
                "prov_has_country_compact",
                "prov_has_token_shingles",
                "prov_has_addr_num_stem",
                "prov_has_postal_num",
                "prov_has_postal_prefix",
            ])

        if self.enable_cross_source_features:
            features.extend([
                "target_source_is_s2",
                "target_source_is_s3",
                "pair_type_s1_s2",
                "pair_type_s1_s3",
            ])

        return features
