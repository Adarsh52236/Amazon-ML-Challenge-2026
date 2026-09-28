"""Unit tests for matching models, label building, validation, and training routines."""

import numpy as np
import pandas as pd
import pytest

from business_entity_resolution.models.label_builder import LabelBuilder, LabeledDataset
from business_entity_resolution.models.model_config import (
    LightGBMConfig,
    LogisticRegressionConfig,
    ModelConfig,
    ValidationConfig,
    XGBoostConfig,
)
from business_entity_resolution.models.train import ModelTrainer
from business_entity_resolution.models.validation import Evaluator, ValidationResults


def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.validation.n_splits == 5
    assert cfg.validation.random_state == 42
    assert len(cfg.validation.threshold_grid) == 19
    assert cfg.validation.threshold_grid[0] == pytest.approx(0.05)
    assert cfg.validation.threshold_grid[-1] == pytest.approx(0.95)
    assert cfg.logistic_regression.C == 1.0
    assert cfg.lightgbm.n_estimators == 300
    assert cfg.xgboost.tree_method == "hist"


def test_label_builder_synthetic():
    features_data = {
        "source1_entity_id": ["S1-1", "S1-1", "S1-2", "S1-3"],
        "target_entity_id": ["T-1", "T-2", "T-3", "T-4"],
        "target_source": ["s2", "s3", "s2", "s3"],
        "f1": [0.9, 0.1, 0.8, 0.2],
        "f2": [1.0, 0.0, 0.7, 0.0],
    }
    df = pd.DataFrame(features_data)
    gt_dict = {
        "S1-1": ["T-1"],
        "S1-2": ["T-3"],
    }
    builder = LabelBuilder(ground_truth_dict=gt_dict)
    labeled = builder.build_labeled_dataset(df, strict_audit=False)

    assert labeled.n_samples == 4
    assert labeled.n_features == 2
    assert labeled.feature_names == ["f1", "f2"]
    assert labeled.n_positives == 2
    assert labeled.n_negatives == 2
    assert list(labeled.y) == [1, 0, 1, 0]


def test_label_builder_duplicate_error():
    features_data = {
        "source1_entity_id": ["S1-1", "S1-1"],
        "target_entity_id": ["T-1", "T-1"],
        "target_source": ["s2", "s2"],
        "f1": [0.9, 0.9],
    }
    df = pd.DataFrame(features_data)
    builder = LabelBuilder(ground_truth_dict={})
    with pytest.raises(ValueError, match="duplicate"):
        builder.build_labeled_dataset(df, strict_audit=False)


def test_evaluator_group_disjointness():
    groups = np.array(["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"])
    X = pd.DataFrame({"f": np.arange(10)})
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    val_cfg = ValidationConfig(n_splits=5)
    evaluator = Evaluator(val_cfg)
    folds = evaluator.get_group_folds(X, y, groups)

    assert len(folds) == 5
    for train_idx, val_idx in folds:
        train_grp = set(groups[train_idx])
        val_grp = set(groups[val_idx])
        assert len(train_grp.intersection(val_grp)) == 0


def test_evaluator_threshold_computation():
    val_cfg = ValidationConfig(
        n_splits=2,
        threshold_min=0.2,
        threshold_max=0.8,
        threshold_steps=4,
        candidate_ground_truth_pairs=2,
        total_ground_truth_pairs=4,
    )
    evaluator = Evaluator(val_cfg)
    y_true = np.array([1, 1, 0, 0])
    oof_probs = np.array([0.9, 0.3, 0.4, 0.1])

    res = evaluator.evaluate_probabilities("TestModel", y_true, oof_probs, 0.5)
    assert res.model_name == "TestModel"
    assert res.pr_auc > 0.8
    assert res.best_candidate_f1 > 0.0
    assert len(res.threshold_metrics) == 4

    # At threshold 0.8, only the first sample (0.9) is predicted positive: TP=1, FP=0
    # Candidate recall = 1/2 = 0.5, Precision = 1.0, F1 = 2/3 = 0.667
    # Competition recall = 1/4 = 0.25, F1 = 2*(1.0*0.25)/(1.25) = 0.4
    t_08 = [m for m in res.threshold_metrics if abs(m.threshold - 0.8) < 1e-4][0]
    assert t_08.true_positives == 1
    assert t_08.false_positives == 0
    assert t_08.candidate_precision == pytest.approx(1.0)
    assert t_08.candidate_recall == pytest.approx(0.5)
    assert t_08.competition_recall == pytest.approx(0.25)


def test_trainer_end_to_end_fast():
    # Build synthetic separable dataset
    np.random.seed(42)
    n = 100
    groups = np.repeat(np.arange(10), 10).astype(str)
    y = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    X = pd.DataFrame({
        "f1": y * 2.0 + np.random.randn(n) * 0.2,
        "f2": y * 1.5 + np.random.randn(n) * 0.3,
    })
    trace_df = pd.DataFrame({
        "source1_entity_id": groups,
        "target_entity_id": [f"T-{i}" for i in range(n)],
        "target_source": ["s2"] * n,
    })

    dataset = LabeledDataset(
        X=X,
        y=y,
        groups=groups,
        trace_df=trace_df,
        feature_names=["f1", "f2"],
    )

    cfg = ModelConfig()
    cfg.validation.n_splits = 3
    cfg.validation.candidate_ground_truth_pairs = int(y.sum())
    cfg.validation.total_ground_truth_pairs = int(y.sum()) + 5
    cfg.logistic_regression.max_iter = 100
    cfg.lightgbm.n_estimators = 20
    cfg.xgboost.n_estimators = 20

    trainer = ModelTrainer(cfg)

    # 1. Test LR
    lr_res, lr_meta = trainer.train_logistic_regression(dataset)
    assert lr_res.pr_auc > 0.7
    assert len(lr_res.oof_probabilities) == n
    assert "f1" in lr_meta["feature_coefficients"]

    # 2. Test LightGBM
    lgb_res, lgb_meta = trainer.train_lightgbm(dataset)
    assert lgb_res.pr_auc > 0.7
    assert len(lgb_res.oof_probabilities) == n
    assert "feature_importance_gain" in lgb_meta

    # 3. Test XGBoost
    xgb_res, xgb_meta = trainer.train_xgboost(dataset)
    assert xgb_res.pr_auc > 0.7
    assert len(xgb_res.oof_probabilities) == n
