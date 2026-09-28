"""Configuration dataclasses for model training, tuning, and validation."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class ValidationConfig:
    """Cross-validation and evaluation parameters."""

    n_splits: int = 5
    random_state: int = 42
    group_col: str = "source1_entity_id"
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_steps: int = 19
    total_ground_truth_pairs: int = 17301
    candidate_ground_truth_pairs: int = 14862

    @property
    def threshold_grid(self) -> np.ndarray:
        return np.linspace(
            self.threshold_min, self.threshold_max, self.threshold_steps
        )


@dataclass
class LogisticRegressionConfig:
    """Hyperparameters for Logistic Regression baseline."""

    C: float = 1.0
    penalty: str = "l2"
    solver: str = "lbfgs"
    max_iter: int = 1000
    random_state: int = 42
    scale_features: bool = True


@dataclass
class LightGBMConfig:
    """Hyperparameters for LightGBM classifier."""

    objective: str = "binary"
    metric: str = "binary_logloss"
    boosting_type: str = "gbdt"
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = -1
    early_stopping_rounds: int = 30


@dataclass
class XGBoostConfig:
    """Hyperparameters for XGBoost classifier."""

    objective: str = "binary:logistic"
    eval_metric: str = "logloss"
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    n_jobs: int = -1
    tree_method: str = "hist"
    early_stopping_rounds: int = 30


@dataclass
class ModelConfig:
    """Master model and validation configuration container."""

    validation: ValidationConfig = field(default_factory=ValidationConfig)
    logistic_regression: LogisticRegressionConfig = field(
        default_factory=LogisticRegressionConfig
    )
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    trace_columns: List[str] = field(
        default_factory=lambda: [
            "source1_entity_id",
            "target_entity_id",
            "target_source",
        ]
    )
