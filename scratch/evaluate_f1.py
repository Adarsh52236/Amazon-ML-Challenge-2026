import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_validate, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from business_entity_resolution.models.label_builder import LabelBuilder
from business_entity_resolution.data_loader import load_ground_truth

def main():
    feat_path = "data/processed/features_dev_phase5_improved.parquet"
    print(f"Loading {feat_path}...")
    df = pd.read_parquet(feat_path)
    gt = load_ground_truth(as_dict=True)

    lb = LabelBuilder()
    dataset = lb.build_labeled_dataset(df, gt)
    
    print(f"Dataset shape: {dataset.X.shape}, Positives: {dataset.y.sum()}")
    
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
    
    scoring = {
        'f1': make_scorer(f1_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score)
    }
    
    from sklearn.model_selection import cross_val_predict
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("Evaluating XGBoost with cross_val_predict...")
    y_probs = cross_val_predict(xgb_model, dataset.X, dataset.y, cv=cv, method='predict_proba', n_jobs=1)[:, 1]
    
    actual_positives = 17301
    
    best_f1 = 0
    best_thresh = 0
    best_prec = 0
    best_rec = 0
    
    for thresh in np.arange(0.5, 1.0, 0.05):
        y_pred = (y_probs >= thresh).astype(int)
        
        tp = np.sum((y_pred == 1) & (dataset.y == 1))
        fp = np.sum((y_pred == 1) & (dataset.y == 0))
        fn = actual_positives - tp
        
        comp_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        comp_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        comp_rec = tp / actual_positives
        
        comp_f05 = (1.25 * comp_prec * comp_rec) / (0.25 * comp_prec + comp_rec) if (0.25 * comp_prec + comp_rec) > 0 else 0
        print(f"Thresh {thresh:.2f} | F0.5: {comp_f05:.4f} | Prec: {comp_prec:.4f} | Rec: {comp_rec:.4f} | FP: {fp} | TP: {tp}")
        
        if comp_f05 > best_f1:
            best_f1 = comp_f05
            best_thresh = thresh
            best_prec = comp_prec
            best_rec = comp_rec
            
    print(f"\nOptimal Competition Metrics (Threshold {best_thresh:.2f}):")
    print(f"  Competition F0.5:      {best_f1:.6f}")
    print(f"  Competition Precision: {best_prec:.6f}")
    print(f"  Competition Recall:    {best_rec:.6f}")
    
if __name__ == "__main__":
    main()
