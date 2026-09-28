import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from business_entity_resolution.preprocessing_config import BlockingConfig
from business_entity_resolution.candidate_generator import CandidateGenerator
from business_entity_resolution.data_loader import load_ground_truth, load_source_file, iter_source_records, get_source_path
from business_entity_resolution.features.feature_config import FeatureConfig
from business_entity_resolution.features.pairwise_extractor import PairwiseFeatureExtractor

def evaluate_fold_preds(preds_df, gt_dict, threshold, name="Model"):
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
            f05 = 1.0 if len(gt_set) == 0 else 0.0
        else:
            prec = tp / (tp + fp) if tp + fp > 0 else 0.0
            rec = tp / (tp + fn) if tp + fn > 0 else 0.0
            f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if prec > 0 or rec > 0 else 0.0
                
        f_scores.append(f05)
        entity_stats[s1_id] = {'tp': tp, 'fp': fp, 'fn': fn, 'f05': f05, 'gt_len': len(gt_set)}
        
    macro_f05 = np.mean(f_scores)
    macro_f05_std = np.std(f_scores)
    
    zm_entities = [s for s in all_s1_ids if entity_stats[s]['gt_len'] == 0]
    zm_fp_rate = sum(1 for s in zm_entities if entity_stats[s]['fp'] > 0) / max(1, len(zm_entities))
    
    sg_entities = [s for s in all_s1_ids if entity_stats[s]['gt_len'] == 1]
    sg_acc = sum(1 for s in sg_entities if entity_stats[s]['tp'] == 1 and entity_stats[s]['fp'] == 0) / max(1, len(sg_entities))
    
    return {
        'Macro_F05': macro_f05,
        'Macro_F05_std': macro_f05_std,
        'ZM_FP_Rate': zm_fp_rate,
        'Singleton_Acc': sg_acc,
        'Precision': tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else 0.0,
        'Recall': tp_total / (tp_total + fn_total) if tp_total + fn_total > 0 else 0.0,
        'TP': tp_total,
        'FP': fp_total,
        'FN': fn_total,
        'preds_dict': dict(preds_dict)
    }

def main():
    seed = 42
    sample_size = 5000
    target_scan_limit = 100_000

    print("Loading data...")
    gt_mapping = load_ground_truth(as_dict=True)
    s1_all = load_source_file("train", 1)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(len(s1_all), size=sample_size, replace=False)
    s1_dev = s1_all.iloc[sampled].copy().reset_index(drop=True)
    dev_s1_ids = set(s1_dev["entity_id"])

    dev_gt = {s1_id: set(gt_mapping.get(s1_id, [])) for s1_id in dev_s1_ids if s1_id in gt_mapping}
    all_gt_pairs = set()
    for s1_id, targets in dev_gt.items():
        for t in targets:
            all_gt_pairs.add((s1_id, t))
    target_ids_needed = {t for _, t in all_gt_pairs}

    s2_path = get_source_path("train", 2)
    s3_path = get_source_path("train", 3)
    raw_targets_data = {}
    indexed_s2 = 0
    for r in iter_source_records(s2_path):
        eid = r["entity_id"]
        if eid in target_ids_needed or indexed_s2 < target_scan_limit:
            raw_targets_data[eid] = {"name": r.get("business_name", ""), "address": r.get("business_address", ""), "country": r.get("country", ""), "source": "S2"}
            indexed_s2 += 1
            
    indexed_s3 = 0
    for r in iter_source_records(s3_path):
        eid = r["entity_id"]
        if eid in target_ids_needed or indexed_s3 < target_scan_limit:
            raw_targets_data[eid] = {"name": r.get("business_name", ""), "address": r.get("business_address", ""), "country": r.get("country", ""), "source": "S3"}
            indexed_s3 += 1

    print("Generating candidates...")
    config_uncapped = BlockingConfig(
        use_stem_prefix=True, use_transliteration=True, use_rare_name_token=True,
        use_addr_alpha=True, max_candidates_per_entity=10000, max_key_frequency=5000
    )
    config_uncapped.channel_weights = {}
    generator = CandidateGenerator(config=config_uncapped)
    
    indexed_s2 = 0
    for r in iter_source_records(s2_path):
        eid = r["entity_id"]
        if eid in target_ids_needed or indexed_s2 < target_scan_limit:
            generator.index_target_record(eid, r.get("business_name", ""), r.get("business_address", ""), r.get("country", ""))
            indexed_s2 += 1
    indexed_s3 = 0
    for r in iter_source_records(s3_path):
        eid = r["entity_id"]
        if eid in target_ids_needed or indexed_s3 < target_scan_limit:
            generator.index_target_record(eid, r.get("business_name", ""), r.get("business_address", ""), r.get("country", ""))
            indexed_s3 += 1
            
    generator.prune_high_frequency_keys()
    
    uncapped_cands_list = generator.generate_candidates_for_dataframe(s1_dev)
    
    s1_cands_ordered = defaultdict(list)
    for c in uncapped_cands_list:
        s1_cands_ordered[c.source1_entity_id].append(c)

    capped_cands = []
    for s1_id in s1_dev['entity_id']:
        s1_id = str(s1_id)
        cands = s1_cands_ordered.get(s1_id, [])
        cands.sort(key=lambda c: len(c.channels), reverse=True)
        capped_cands.extend(cands[:100])

    print("Extracting features...")
    feat_cfg = FeatureConfig()
    extractor = PairwiseFeatureExtractor(config=feat_cfg)
    
    s1_preprocessed = {}
    for _, row in s1_dev.iterrows():
        eid = str(row["entity_id"])
        s1_preprocessed[eid] = extractor.preprocess_entity(eid, row.get("business_name", ""), row.get("business_address", ""), row.get("country", ""), "S1")
    
    target_preprocessed = {}
    for tid, r in raw_targets_data.items():
        target_preprocessed[tid] = extractor.preprocess_entity(tid, r["name"], r["address"], r["country"], r["source"])
    
    feature_df = extractor.extract_features_for_candidates(capped_cands, s1_preprocessed, target_preprocessed)
    
    labels = []
    for s1, t in zip(feature_df['source1_entity_id'], feature_df['target_entity_id']):
        labels.append(1 if t in dev_gt.get(s1, set()) else 0)
    feature_df['label'] = labels
    
    # Identify new features
    new_features = ["target_address_missing", "name_strong_target_address_missing", "name_consonant_strong_target_address_missing"]
    base_cols = [c for c in feature_df.columns if c not in ["source1_entity_id", "target_entity_id", "target_source", "label"] + new_features]
    new_cols = base_cols + new_features
    
    X_base = feature_df[base_cols].values
    X_new = feature_df[new_cols].values
    y = feature_df['label'].values
    groups = feature_df['source1_entity_id'].values
    
    gkf = GroupKFold(n_splits=5)
    
    xgb_params = {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8, 'tree_method': 'hist', 'n_jobs': -1, 'random_state': 42}
    lgb_params = {'objective': 'binary', 'metric': 'binary_logloss', 'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'verbose': -1}
    
    base_preds = []
    new_preds = []
    
    print("Training baseline models...")
    for train_idx, val_idx in gkf.split(X_base, y, groups):
        X_train, X_val = X_base[train_idx], X_base[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        m_xgb = xgb.XGBClassifier(**xgb_params)
        m_xgb.fit(X_train, y_train)
        p_xgb = m_xgb.predict_proba(X_val)[:, 1]
        
        m_lgb = lgb.LGBMClassifier(**lgb_params)
        m_lgb.fit(X_train, y_train)
        p_lgb = m_lgb.predict_proba(X_val)[:, 1]
        
        p_blend = 0.75 * p_xgb + 0.25 * p_lgb
        
        val_df = feature_df.iloc[val_idx].copy()
        val_df['prob'] = p_blend
        base_preds.append(val_df)
        
    print("Training models with conditional features...")
    # Get feature importances from last fold
    new_fi_xgb, new_fi_lgb = None, None
    for train_idx, val_idx in gkf.split(X_new, y, groups):
        X_train, X_val = X_new[train_idx], X_new[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        m_xgb = xgb.XGBClassifier(**xgb_params)
        m_xgb.fit(X_train, y_train)
        p_xgb = m_xgb.predict_proba(X_val)[:, 1]
        
        m_lgb = lgb.LGBMClassifier(**lgb_params)
        m_lgb.fit(X_train, y_train)
        p_lgb = m_lgb.predict_proba(X_val)[:, 1]
        
        p_blend = 0.75 * p_xgb + 0.25 * p_lgb
        
        val_df = feature_df.iloc[val_idx].copy()
        val_df['prob'] = p_blend
        new_preds.append(val_df)
        
        new_fi_xgb = m_xgb.feature_importances_
        new_fi_lgb = m_lgb.feature_importances_
        
    df_base = pd.concat(base_preds, ignore_index=True)
    df_new = pd.concat(new_preds, ignore_index=True)
    
    df_base = df_base.merge(s1_dev[['entity_id', 'country']], left_on='source1_entity_id', right_on='entity_id')
    df_new = df_new.merge(s1_dev[['entity_id', 'country']], left_on='source1_entity_id', right_on='entity_id')
    
    res_base = evaluate_fold_preds(df_base, dev_gt, 0.75, "Baseline")
    res_new = evaluate_fold_preds(df_new, dev_gt, 0.75, "New")
    
    res_base_in = evaluate_fold_preds(df_base[df_base['country'] == 'India'], dev_gt, 0.75)
    res_base_us = evaluate_fold_preds(df_base[df_base['country'] == 'US'], dev_gt, 0.75)
    res_new_in = evaluate_fold_preds(df_new[df_new['country'] == 'India'], dev_gt, 0.75)
    res_new_us = evaluate_fold_preds(df_new[df_new['country'] == 'US'], dev_gt, 0.75)
    
    # Missing address subset analysis
    missing_addr_mask = feature_df['target_address_missing'] == 1.0
    missing_gt_pairs = set(zip(feature_df[missing_addr_mask & (feature_df['label']==1)]['source1_entity_id'], feature_df[missing_addr_mask & (feature_df['label']==1)]['target_entity_id']))
    
    base_tp_missing = 0
    new_tp_missing = 0
    base_fp_missing = 0
    new_fp_missing = 0
    
    # Helper to count TP/FP in a subset
    def count_subset(preds_df, mask):
        tp = fp = 0
        df_sub = preds_df[mask & (preds_df['prob'] >= 0.75)]
        for _, r in df_sub.iterrows():
            if r['label'] == 1: tp += 1
            else: fp += 1
        return tp, fp
        
    base_tp_missing, base_fp_missing = count_subset(df_base, missing_addr_mask)
    new_tp_missing, new_fp_missing = count_subset(df_new, missing_addr_mask)
    
    missing_gt_len = len(missing_gt_pairs)
    base_fn_missing = missing_gt_len - base_tp_missing
    new_fn_missing = missing_gt_len - new_tp_missing
    
    # Global changes
    global_lost_tp = max(0, res_base['TP'] - res_new['TP']) # net lost TP? No, let's do exact sets
    base_tps = set(zip(df_base[(df_base['prob']>=0.75) & (df_base['label']==1)]['source1_entity_id'], df_base[(df_base['prob']>=0.75) & (df_base['label']==1)]['target_entity_id']))
    new_tps = set(zip(df_new[(df_new['prob']>=0.75) & (df_new['label']==1)]['source1_entity_id'], df_new[(df_new['prob']>=0.75) & (df_new['label']==1)]['target_entity_id']))
    
    lost_tp_set = base_tps - new_tps
    gained_tp_set = new_tps - base_tps
    
    base_fps = set(zip(df_base[(df_base['prob']>=0.75) & (df_base['label']==0)]['source1_entity_id'], df_base[(df_base['prob']>=0.75) & (df_base['label']==0)]['target_entity_id']))
    new_fps = set(zip(df_new[(df_new['prob']>=0.75) & (df_new['label']==0)]['source1_entity_id'], df_new[(df_new['prob']>=0.75) & (df_new['label']==0)]['target_entity_id']))
    
    gained_fp_set = new_fps - base_fps
    
    # Print results
    print(f"BASELINE: Macro F0.5 = {res_base['Macro_F05']:.4f} ± {res_base['Macro_F05_std']:.4f}")
    print(f"NEW:      Macro F0.5 = {res_new['Macro_F05']:.4f} ± {res_new['Macro_F05_std']:.4f}")
    
    import json
    results = {
        "baseline": {
            "macro_f05": res_base['Macro_F05'],
            "macro_f05_std": res_base['Macro_F05_std'],
            "precision": res_base['Precision'],
            "recall": res_base['Recall'],
            "tp": res_base['TP'],
            "fp": res_base['FP'],
            "fn": res_base['FN'],
            "zm_fp": res_base['ZM_FP_Rate'],
            "singleton_acc": res_base['Singleton_Acc'],
            "in_macro_f05": res_base_in['Macro_F05'],
            "us_macro_f05": res_base_us['Macro_F05']
        },
        "new": {
            "macro_f05": res_new['Macro_F05'],
            "macro_f05_std": res_new['Macro_F05_std'],
            "precision": res_new['Precision'],
            "recall": res_new['Recall'],
            "tp": res_new['TP'],
            "fp": res_new['FP'],
            "fn": res_new['FN'],
            "zm_fp": res_new['ZM_FP_Rate'],
            "singleton_acc": res_new['Singleton_Acc'],
            "in_macro_f05": res_new_in['Macro_F05'],
            "us_macro_f05": res_new_us['Macro_F05']
        },
        "missing_subset": {
            "gt_pairs": missing_gt_len,
            "base_tp": base_tp_missing,
            "base_fn": base_fn_missing,
            "new_tp": new_tp_missing,
            "new_fn": new_fn_missing,
            "fn_recovered": base_fn_missing - new_fn_missing,
            "new_fp": new_fp_missing - base_fp_missing
        },
        "global_effect": {
            "baseline_tps_lost": len(lost_tp_set),
            "total_fns_recovered": len(gained_tp_set),
            "new_fps": len(gained_fp_set)
        },
        "feature_importance": {
            "xgb": {col: float(imp) for col, imp in zip(new_cols, new_fi_xgb) if col in new_features},
            "lgb": {col: float(imp) for col, imp in zip(new_cols, new_fi_lgb) if col in new_features}
        }
    }
    
    with open('output/phase16_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
