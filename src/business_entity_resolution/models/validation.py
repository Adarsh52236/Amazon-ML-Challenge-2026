"""Validation and threshold evaluation module for matching models."""

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from business_entity_resolution.models.model_config import ValidationConfig


@dataclass
class ThresholdMetrics:
    """Detailed evaluation metrics at a specific decision threshold."""

    threshold: float
    predicted_positive: int
    candidate_precision: float
    candidate_recall: float
    candidate_f1: float
    competition_recall: float
    competition_f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int


@dataclass
class ValidationResults:
    """Summary of out-of-fold validation and threshold grid search."""

    model_name: str
    pr_auc: float
    roc_auc: float
    training_runtime_seconds: float
    best_candidate_threshold: float
    best_candidate_f1: float
    best_competition_threshold: float
    best_competition_f1: float
    threshold_metrics: List[ThresholdMetrics]
    oof_probabilities: np.ndarray


class Evaluator:
    """Evaluates cross-validation performance, threshold tuning, and entity-level metrics."""

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()

    def get_group_folds(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate strictly entity-disjoint folds via GroupKFold."""
        gkf = GroupKFold(n_splits=self.config.n_splits)
        folds = list(gkf.split(X, y, groups=groups))

        # Strict leakage verification: no group can appear in both train and val
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            train_groups = set(groups[train_idx])
            val_groups = set(groups[val_idx])
            overlap = train_groups.intersection(val_groups)
            if overlap:
                raise ValueError(
                    f"Leakage detected in fold {fold_idx}: {len(overlap)} groups overlap!"
                )

        return folds

    def evaluate_probabilities(
        self,
        model_name: str,
        y_true: np.ndarray,
        oof_probs: np.ndarray,
        runtime_seconds: float,
    ) -> ValidationResults:
        """Evaluate threshold grid across candidate-set and overall competition denominators."""
        pr_auc = float(average_precision_score(y_true, oof_probs))
        roc_auc = float(roc_auc_score(y_true, oof_probs))

        threshold_metrics: List[ThresholdMetrics] = []
        best_cand_f1 = 0.0
        best_cand_thresh = 0.5
        best_comp_f1 = 0.0
        best_comp_thresh = 0.5

        cand_denom = self.config.candidate_ground_truth_pairs
        comp_denom = self.config.total_ground_truth_pairs

        for t in self.config.threshold_grid:
            preds = (oof_probs >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
            pred_pos = int(tp + fp)

            p = float(tp / pred_pos) if pred_pos > 0 else 0.0
            r_cand = float(tp / cand_denom)
            f1_cand = (
                float(2 * p * r_cand / (p + r_cand)) if (p + r_cand) > 0 else 0.0
            )

            r_comp = float(tp / comp_denom)
            f1_comp = (
                float(2 * p * r_comp / (p + r_comp)) if (p + r_comp) > 0 else 0.0
            )

            if f1_cand > best_cand_f1:
                best_cand_f1 = f1_cand
                best_cand_thresh = float(t)

            if f1_comp > best_comp_f1:
                best_comp_f1 = f1_comp
                best_comp_thresh = float(t)

            threshold_metrics.append(
                ThresholdMetrics(
                    threshold=float(t),
                    predicted_positive=pred_pos,
                    candidate_precision=p,
                    candidate_recall=r_cand,
                    candidate_f1=f1_cand,
                    competition_recall=r_comp,
                    competition_f1=f1_comp,
                    true_positives=int(tp),
                    false_positives=int(fp),
                    false_negatives=int(fn),
                    true_negatives=int(tn),
                )
            )

        return ValidationResults(
            model_name=model_name,
            pr_auc=pr_auc,
            roc_auc=roc_auc,
            training_runtime_seconds=runtime_seconds,
            best_candidate_threshold=best_cand_thresh,
            best_candidate_f1=best_cand_f1,
            best_competition_threshold=best_comp_thresh,
            best_competition_f1=best_comp_f1,
            threshold_metrics=threshold_metrics,
            oof_probabilities=oof_probs,
        )
