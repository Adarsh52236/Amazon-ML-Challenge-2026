import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from collections import defaultdict
import json

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from business_entity_resolution.data_loader import load_ground_truth

def evaluate_fold_preds(preds_df, gt_dict, threshold):
    preds_dict = defaultdict(list)
    mask = preds_df['prob'] >= threshold
    for s1_id, t_id in zip(preds_df[mask]['source1_entity_id'], preds_df[mask]['target_entity_id']):
        preds_dict[s1_id].append(t_id)
        
    f_scores = []
    tp_total = fp_total = fn_total = 0
    all_s1_ids = preds_df['source1_entity_id'].unique()
    
    entity_stats = {}
    
    for s1_id in all_s1_ids:
        gt_set = set(gt_dict.get(s1_id, []))
        pred_set = set(preds_dict.get(s1_id, []))
        
        tp = len(gt_set & pred_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        tp_total += tp
        fp_total += fp
        fn_total += fn
        
        if tp == 0 and fp == 0:
            if len(gt_set) == 0:
                f05 = 1.0
            else:
                f05 = 0.0
        else:
            prec = tp / (tp + fp) if tp + fp > 0 else 0.0
            rec = tp / (tp + fn) if tp + fn > 0 else 0.0
            if prec > 0 or rec > 0:
                f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
            else:
                f05 = 0.0
                
        f_scores.append(f05)
        entity_stats[s1_id] = {'tp': tp, 'fp': fp, 'fn': fn, 'f05': f05, 'gt_len': len(gt_set)}
        
    macro_f05 = np.mean(f_scores)
    
    g_prec = tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else 0.0
    g_rec = tp_total / (tp_total + fn_total) if tp_total + fn_total > 0 else 0.0
    g_f1 = (2 * g_prec * g_rec) / (g_prec + g_rec) if g_prec + g_rec > 0 else 0.0
    
    zm_entities = [s for s in all_s1_ids if entity_stats[s]['gt_len'] == 0]
    zm_fp_rate = sum(1 for s in zm_entities if entity_stats[s]['fp'] > 0) / len(zm_entities) if zm_entities else 0.0
    
    sg_entities = [s for s in all_s1_ids if entity_stats[s]['gt_len'] == 1]
    sg_acc = sum(1 for s in sg_entities if entity_stats[s]['tp'] == 1 and entity_stats[s]['fp'] == 0) / len(sg_entities) if sg_entities else 0.0
    
    mm_entities = [s for s in all_s1_ids if entity_stats[s]['gt_len'] > 1]
    mm_f05 = np.mean([entity_stats[s]['f05'] for s in mm_entities]) if mm_entities else 0.0
    
    fully_rec = sum(1 for s in all_s1_ids if entity_stats[s]['gt_len'] > 0 and entity_stats[s]['fn'] == 0) / (len(all_s1_ids) - len(zm_entities))
    part_rec = sum(1 for s in all_s1_ids if entity_stats[s]['gt_len'] > 0 and entity_stats[s]['tp'] > 0 and entity_stats[s]['fn'] > 0) / (len(all_s1_ids) - len(zm_entities))
    miss_rec = sum(1 for s in all_s1_ids if entity_stats[s]['gt_len'] > 0 and entity_stats[s]['tp'] == 0) / (len(all_s1_ids) - len(zm_entities))
    
    return {
        'Macro_F05': macro_f05,
        'Global_F1': g_f1,
        'Global_Prec': g_prec,
        'Global_Rec': g_rec,
        'ZM_FP_Rate': zm_fp_rate,
        'Singleton_Acc': sg_acc,
        'MultiMatch_F05': mm_f05,
        'Fully_Rec': fully_rec,
        'Part_Rec': part_rec,
        'Miss_Rec': miss_rec
    }

def main():
    feat_path = ROOT / "data/processed/features_dev_phase5_improved.parquet"
    df = pd.read_parquet(feat_path)
    gt_dict = load_ground_truth(as_dict=True)
    
    # Build labels
    labels = []
    for s1, t in zip(df['source1_entity_id'], df['target_entity_id']):
        labels.append(1 if t in gt_dict.get(s1, set()) else 0)
    df['label'] = labels
    
    feature_cols = [c for c in df.columns if c not in ["source1_entity_id", "target_entity_id", "target_source", "label"]]
    X = df[feature_cols].values
    y = df['label'].values
    groups = df['source1_entity_id'].values
    
    gkf = GroupKFold(n_splits=5)
    
    all_results = []
    
    xgb_params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'n_jobs': -1,
        'random_state': 42
    }
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'n_estimators': 300,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    fold = 1
    for train_idx, val_idx in gkf.split(X, y, groups):
        print(f"Fold {fold}...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train XGB
        model_xgb = xgb.XGBClassifier(**xgb_params)
        model_xgb.fit(X_train, y_train)
        p_xgb = model_xgb.predict_proba(X_val)[:, 1]
        
        # Train LGB
        model_lgb = lgb.LGBMClassifier(**lgb_params)
        model_lgb.fit(X_train, y_train)
        p_lgb = model_lgb.predict_proba(X_val)[:, 1]
        
        p_blend = 0.75 * p_xgb + 0.25 * p_lgb
        
        val_df = df.iloc[val_idx].copy()
        
        thresholds = np.arange(0.30, 0.85, 0.05)
        for thresh in thresholds:
            for model_name, probs in [('XGB', p_xgb), ('LGB', p_lgb), ('Blend', p_blend)]:
                val_df['prob'] = probs
                metrics = evaluate_fold_preds(val_df, gt_dict, thresh)
                metrics['Fold'] = fold
                metrics['Model'] = model_name
                metrics['Threshold'] = round(float(thresh), 2)
                all_results.append(metrics)
        fold += 1
        
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(ROOT / "output/oos_cv_results.tsv", sep="\t", index=False)
    
if __name__ == "__main__":
    main()
