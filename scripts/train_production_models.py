"""
Train production XGBoost and LightGBM models on the verified Phase 5 development feature matrix.
"""

import os
import time
from pathlib import Path
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from business_entity_resolution.data_loader import load_ground_truth
from business_entity_resolution.models.label_builder import LabelBuilder

def main():
    t0 = time.time()
    print("=" * 80)
    print("PHASE 6: TRAINING PRODUCTION MODELS")
    print("=" * 80)

    feat_path = "data/processed/features_dev_phase5_improved.parquet"
    print(f"Loading development feature matrix from {feat_path}...")
    df = pd.read_parquet(feat_path)
    gt = load_ground_truth(as_dict=True)

    lb = LabelBuilder()
    dataset = lb.build_labeled_dataset(df, gt)
    print(f"Dataset: {dataset.n_samples:,} rows, {dataset.n_features} features, {dataset.n_positives:,} positives")

    Path("models").mkdir(exist_ok=True)

    # 1. Fit XGBoost
    print("Training production XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        n_jobs=-1
    )
    xgb_model.fit(dataset.X, dataset.y)
    xgb_path = "models/model_xgb_prod.json"
    xgb_model.save_model(xgb_path)
    print(f"XGBoost saved to {xgb_path} ({os.path.getsize(xgb_path) / 1024:.1f} KB)")

    # 2. Fit LightGBM
    print("Training production LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        boosting_type="gbdt",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(dataset.X, dataset.y)
    lgb_path = "models/model_lgb_prod.txt"
    lgb_model.booster_.save_model(lgb_path)
    print(f"LightGBM saved to {lgb_path} ({os.path.getsize(lgb_path) / 1024:.1f} KB)")

    print(f"Production models successfully trained and serialized in {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()
