"""Pairwise feature extraction orchestrator for Business Entity Resolution.

Extracts structured, deterministic, leakage-free feature vectors comparing
Source 1 reference entities against secondary candidate entities (Source 2 / Source 3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from business_entity_resolution.address_normalizer import (
    BusinessAddressNormalizer,
    NormalizedBusinessAddress,
)
from business_entity_resolution.candidate_generator import CandidatePair
from business_entity_resolution.features.address_features import extract_address_features
from business_entity_resolution.features.country_features import extract_country_features
from business_entity_resolution.features.feature_config import FeatureConfig
from business_entity_resolution.features.name_features import extract_name_features
from business_entity_resolution.features.provenance_features import extract_provenance_features
from business_entity_resolution.features.source_features import extract_source_features
from business_entity_resolution.text_normalizer import (
    BusinessNameNormalizer,
    NormalizedBusinessName,
)

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedEntity:
    """Pre-normalized entity representation for O(1) pairwise comparison lookups."""

    entity_id: str
    norm_name: NormalizedBusinessName
    norm_address: NormalizedBusinessAddress
    country: str
    source: str


class PairwiseFeatureExtractor:
    """Orchestrates pairwise feature extraction across candidate pairs."""

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        name_normalizer: Optional[BusinessNameNormalizer] = None,
        addr_normalizer: Optional[BusinessAddressNormalizer] = None,
    ):
        self.config = config or FeatureConfig()
        self.name_normalizer = name_normalizer or BusinessNameNormalizer()
        self.addr_normalizer = addr_normalizer or BusinessAddressNormalizer()
        self.feature_names = self.config.get_feature_names()

    def preprocess_entity(
        self,
        entity_id: str,
        name: Any,
        address: Any,
        country: Any,
        source: Optional[str] = None,
    ) -> PreprocessedEntity:
        """Normalize an entity's name and address once for reuse across multiple candidate pairs."""
        str_name = "" if pd.isna(name) else str(name)
        str_addr = "" if pd.isna(address) else str(address)
        str_country = "" if pd.isna(country) else str(country)

        norm_name = self.name_normalizer.normalize(str_name)
        norm_addr = self.addr_normalizer.normalize(str_addr, country=str_country)

        if source is None:
            if entity_id.startswith("S1-"):
                src = "S1"
            elif entity_id.startswith("S2-"):
                src = "S2"
            elif entity_id.startswith("S3-"):
                src = "S3"
            else:
                src = "UNKNOWN"
        else:
            src = source

        return PreprocessedEntity(
            entity_id=entity_id,
            norm_name=norm_name,
            norm_address=norm_addr,
            country=str_country,
            source=src,
        )

    def extract_features_for_pair(
        self,
        e1: PreprocessedEntity,
        e2: PreprocessedEntity,
        channels: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Extract a single feature vector comparing reference entity e1 and candidate e2."""
        row_features: Dict[str, float] = {}

        # 1. Name Features
        if self.config.enable_name_features:
            row_features.update(extract_name_features(e1.norm_name, e2.norm_name, self.config))

        # 2. Address Features
        if self.config.enable_address_features:
            row_features.update(
                extract_address_features(
                    e1.norm_address, e2.norm_address, country=e1.country, config=self.config
                )
            )

        # 3. Country Features
        if self.config.enable_country_features:
            row_features.update(
                extract_country_features(e1.country, e2.country, self.config)
            )

        # 4. Provenance Features
        if self.config.enable_provenance_features:
            row_features.update(
                extract_provenance_features(channels, self.config)
            )

        # 5. Cross-Source Features
        if self.config.enable_cross_source_features:
            row_features.update(
                extract_source_features(e2.source, self.config)
            )

        return row_features

    def extract_features_for_candidates(
        self,
        candidate_pairs: Sequence[CandidatePair],
        s1_entities: Dict[str, PreprocessedEntity],
        target_entities: Dict[str, PreprocessedEntity],
    ) -> pd.DataFrame:
        """Extract feature matrix for a sequence of candidate pairs.

        Returns
        -------
        pd.DataFrame
            DataFrame with identifiers and ordered feature columns.
            Guaranteed to have zero NaN/inf values.
        """
        rows: List[Dict[str, Any]] = []

        for pair in candidate_pairs:
            s1_id = pair.source1_entity_id
            t_id = pair.matched_entity_id

            e1 = s1_entities.get(s1_id)
            e2 = target_entities.get(t_id)

            if e1 is None:
                raise KeyError(f"Source 1 entity '{s1_id}' not found in preprocessed entities!")
            if e2 is None:
                raise KeyError(f"Target entity '{t_id}' not found in preprocessed entities!")

            features = self.extract_features_for_pair(e1, e2, channels=pair.channels)

            row: Dict[str, Any] = {
                "source1_entity_id": s1_id,
                "target_entity_id": t_id,
                "target_source": pair.matched_source or e2.source,
            }
            # Append features strictly in schema order
            for feat_name in self.feature_names:
                val = features.get(feat_name, self.config.default_missing_numeric)
                # Check for nan/inf
                if np.isnan(val) or np.isinf(val):
                    val = self.config.default_missing_numeric
                row[feat_name] = val

            rows.append(row)

        df = pd.DataFrame(rows)
        # Ensure column ordering: identifiers first, then feature names
        col_order = ["source1_entity_id", "target_entity_id", "target_source"] + self.feature_names
        return df[col_order]
