"""Label construction and validation module for matching models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from business_entity_resolution.data_loader import load_ground_truth


@dataclass
class LabeledDataset:
    """Container holding feature matrix, labels, groups, and trace identifiers."""

    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    trace_df: pd.DataFrame
    feature_names: List[str]

    @property
    def n_samples(self) -> int:
        return len(self.y)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_positives(self) -> int:
        return int((self.y == 1).sum())

    @property
    def n_negatives(self) -> int:
        return int((self.y == 0).sum())


class LabelBuilder:
    """Constructs and validates the binary labeled dataset from features and ground truth."""

    def __init__(
        self,
        trace_columns: Optional[List[str]] = None,
        ground_truth_dict: Optional[Dict[str, List[str]]] = None,
    ):
        self.trace_columns = trace_columns or [
            "source1_entity_id",
            "target_entity_id",
            "target_source",
        ]
        self._gt_dict = ground_truth_dict

    def _get_gt_dict(self) -> Dict[str, Set[str]]:
        if self._gt_dict is None:
            raw_gt = load_ground_truth(as_dict=True)
            self._gt_dict = {s1: set(targets) for s1, targets in raw_gt.items()}
        elif isinstance(next(iter(self._gt_dict.values())), list):
            self._gt_dict = {s1: set(targets) for s1, targets in self._gt_dict.items()}
        return self._gt_dict

    def build_labeled_dataset(
        self,
        features_df: pd.DataFrame,
        strict_audit: bool = True,
    ) -> LabeledDataset:
        """Join features with ground truth and validate integrity."""
        # 1. Check trace columns
        for col in self.trace_columns:
            if col not in features_df.columns:
                raise ValueError(f"Required trace column '{col}' missing from features.")

        # 2. Check duplicate pairs
        dup_count = features_df.duplicated(
            subset=["source1_entity_id", "target_entity_id"]
        ).sum()
        if dup_count > 0:
            raise ValueError(f"Found {dup_count} duplicate candidate pairs.")

        # 3. Separate features from trace
        trace_df = features_df[self.trace_columns].copy()
        feature_cols = [c for c in features_df.columns if c not in self.trace_columns]
        X = features_df[feature_cols].copy()

        # Check for NaN / Inf
        if X.isna().any().any():
            raise ValueError("Feature matrix contains NaN values.")
        if np.isneginf(X.to_numpy()).any() or np.isposinf(X.to_numpy()).any():
            raise ValueError("Feature matrix contains infinite values.")

        # 4. Assign binary label
        gt = self._get_gt_dict()
        s1_arr = trace_df["source1_entity_id"].values
        t_arr = trace_df["target_entity_id"].values

        y = np.zeros(len(features_df), dtype=np.int8)
        for i in range(len(features_df)):
            s1 = s1_arr[i]
            t = t_arr[i]
            if t in gt.get(s1, set()):
                y[i] = 1

        # 5. Strict verification for the official Cap 100 dev matrix
        if strict_audit and len(features_df) == 226217:
            pos_count = int((y == 1).sum())
            neg_count = int((y == 0).sum())
            if pos_count != 14862:
                raise ValueError(
                    f"Label count mismatch: expected 14,862 positives, got {pos_count}"
                )
            if neg_count != 211355:
                raise ValueError(
                    f"Label count mismatch: expected 211,355 negatives, got {neg_count}"
                )

        groups = trace_df["source1_entity_id"].values

        return LabeledDataset(
            X=X,
            y=y,
            groups=groups,
            trace_df=trace_df,
            feature_names=feature_cols,
        )
