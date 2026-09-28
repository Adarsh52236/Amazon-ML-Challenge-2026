"""Model training routines for Business Entity Resolution matching classifiers."""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from business_entity_resolution.models.label_builder import LabeledDataset
from business_entity_resolution.models.model_config import (
    LightGBMConfig,
    LogisticRegressionConfig,
    ModelConfig,
    ValidationConfig,
    XGBoostConfig,
)
from business_entity_resolution.models.validation import Evaluator, ValidationResults


class ModelTrainer:
    """Trains matching models across grouped cross-validation folds."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.evaluator = Evaluator(self.config.validation)

    def train_logistic_regression(
        self,
        dataset: LabeledDataset,
        folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> Tuple[ValidationResults, Dict[str, Any]]:
        """Train regularized Logistic Regression across grouped folds."""
        cfg = self.config.logistic_regression
        if folds is None:
            folds = self.evaluator.get_group_folds(
                dataset.X, dataset.y, dataset.groups
            )

        oof_probs = np.zeros(dataset.n_samples, dtype=np.float32)
        total_time = 0.0
        coefficients = np.zeros(dataset.n_features, dtype=np.float64)

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            t0 = time.time()
            X_train = dataset.X.iloc[train_idx]
            y_train = dataset.y[train_idx]
            X_val = dataset.X.iloc[val_idx]

            if cfg.scale_features:
                scaler = StandardScaler()
                X_train_proc = scaler.fit_transform(X_train)
                X_val_proc = scaler.transform(X_val)
            else:
                X_train_proc = X_train.to_numpy()
                X_val_proc = X_val.to_numpy()

            clf = LogisticRegression(
                C=cfg.C,
                solver=cfg.solver,
                max_iter=cfg.max_iter,
                random_state=cfg.random_state,
            )
            clf.fit(X_train_proc, y_train)
            probs = clf.predict_proba(X_val_proc)[:, 1]
            oof_probs[val_idx] = probs
            coefficients += clf.coef_[0] / len(folds)
            total_time += time.time() - t0

        results = self.evaluator.evaluate_probabilities(
            model_name="Logistic Regression",
            y_true=dataset.y,
            oof_probs=oof_probs,
            runtime_seconds=total_time,
        )

        metadata = {
            "feature_coefficients": dict(zip(dataset.feature_names, coefficients)),
            "scaled": cfg.scale_features,
            "C": cfg.C,
        }
        return results, metadata

    def train_lightgbm(
        self,
        dataset: LabeledDataset,
        folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> Tuple[ValidationResults, Dict[str, Any]]:
        """Train LightGBM gradient boosted trees across grouped folds."""
        import lightgbm as lgb

        cfg = self.config.lightgbm
        if folds is None:
            folds = self.evaluator.get_group_folds(
                dataset.X, dataset.y, dataset.groups
            )

        oof_probs = np.zeros(dataset.n_samples, dtype=np.float32)
        total_time = 0.0
        gain_importances = np.zeros(dataset.n_features, dtype=np.float64)
        split_importances = np.zeros(dataset.n_features, dtype=np.float64)

        params = {
            "objective": cfg.objective,
            "metric": cfg.metric,
            "boosting_type": cfg.boosting_type,
            "n_estimators": cfg.n_estimators,
            "learning_rate": cfg.learning_rate,
            "num_leaves": cfg.num_leaves,
            "max_depth": cfg.max_depth,
            "subsample": cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "random_state": cfg.random_state,
            "n_jobs": cfg.n_jobs,
            "verbose": cfg.verbose,
        }

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            t0 = time.time()
            X_train = dataset.X.iloc[train_idx]
            y_train = dataset.y[train_idx]
            X_val = dataset.X.iloc[val_idx]
            y_val = dataset.y[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=cfg.early_stopping_rounds, verbose=False
                    )
                ],
            )
            probs = model.predict_proba(X_val)[:, 1]
            oof_probs[val_idx] = probs
            gain_importances += (
                model.booster_.feature_importance(importance_type="gain")
                / len(folds)
            )
            split_importances += (
                model.booster_.feature_importance(importance_type="split")
                / len(folds)
            )
            total_time += time.time() - t0

        results = self.evaluator.evaluate_probabilities(
            model_name="LightGBM",
            y_true=dataset.y,
            oof_probs=oof_probs,
            runtime_seconds=total_time,
        )

        metadata = {
            "feature_importance_gain": dict(
                zip(dataset.feature_names, gain_importances)
            ),
            "feature_importance_split": dict(
                zip(dataset.feature_names, split_importances)
            ),
            "hyperparameters": params,
        }
        return results, metadata

    def train_xgboost(
        self,
        dataset: LabeledDataset,
        folds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> Tuple[ValidationResults, Dict[str, Any]]:
        """Train XGBoost gradient boosted trees across grouped folds."""
        import xgboost as xgb

        cfg = self.config.xgboost
        if folds is None:
            folds = self.evaluator.get_group_folds(
                dataset.X, dataset.y, dataset.groups
            )

        oof_probs = np.zeros(dataset.n_samples, dtype=np.float32)
        total_time = 0.0

        params = {
            "objective": cfg.objective,
            "eval_metric": cfg.eval_metric,
            "n_estimators": cfg.n_estimators,
            "learning_rate": cfg.learning_rate,
            "max_depth": cfg.max_depth,
            "subsample": cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "random_state": cfg.random_state,
            "n_jobs": cfg.n_jobs,
            "tree_method": cfg.tree_method,
        }

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            t0 = time.time()
            X_train = dataset.X.iloc[train_idx]
            y_train = dataset.y[train_idx]
            X_val = dataset.X.iloc[val_idx]
            y_val = dataset.y[val_idx]

            model = xgb.XGBClassifier(
                **params, early_stopping_rounds=cfg.early_stopping_rounds
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            probs = model.predict_proba(X_val)[:, 1]
            oof_probs[val_idx] = probs
            total_time += time.time() - t0

        results = self.evaluator.evaluate_probabilities(
            model_name="XGBoost",
            y_true=dataset.y,
            oof_probs=oof_probs,
            runtime_seconds=total_time,
        )

        metadata = {
            "hyperparameters": params,
        }
        return results, metadata
