import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import json

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from business_entity_resolution.preprocessing_config import BlockingConfig
from business_entity_resolution.candidate_generator import CandidateGenerator
from business_entity_resolution.data_loader import load_ground_truth, load_source_file, iter_source_records, get_source_path
from business_entity_resolution.features.feature_config import FeatureConfig
from business_entity_resolution.features.pairwise_extractor import PairwiseFeatureExtractor

def evaluate_fold_preds(preds_df, gt_dict, threshold=0.75):
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
    
    mm_entities = [s for s in all_s1_ids if entity_stats[s]['gt_len'] > 1]
    mm_f_scores = [entity_stats[s]['f05'] for s in mm_entities]
    mm_macro = np.mean(mm_f_scores) if mm_entities else 0.0
    
    return {
        'Macro_F05': macro_f05,
        'Macro_F05_std': macro_f05_std,
        'ZM_FP_Rate': zm_fp_rate,
        'Singleton_Acc': sg_acc,
        'Multi_Macro': mm_macro,
        'Precision': tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else 0.0,
        'Recall': tp_total / (tp_total + fn_total) if tp_total + fn_total > 0 else 0.0,
        'TP': tp_total,
        'FP': fp_total,
        'FN': fn_total,
        'preds_dict': dict(preds_dict),
        'entity_stats': entity_stats
    }

def compute_hardness(df):
    return (
        2.0 * df['name_exact_match'] +
        1.5 * df['prov_has_token_shingles'] +
        1.5 * df['name_token_jaccard'] +
        1.0 * df['name_compact_exact_match'] +
        1.0 * df['address_street_stem_match'] +
        1.0 * df['address_token_jaccard'] +
        1.0 * df['address_street_number_match'] +
        1.0 * df['country_exact_match']
    )

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
    feature_df['hardness'] = compute_hardness(feature_df)
    
    # ----------------------------------------
    # STEP 2: ANALYZE NEGATIVE DISTRIBUTION
    # ----------------------------------------
    all_negatives = feature_df[feature_df['label'] == 0]
    all_positives = feature_df[feature_df['label'] == 1]
    
    neg_hard = all_negatives['hardness'].values
    pos_hard = all_positives['hardness'].values
    
    q1 = np.percentile(neg_hard, 99)
    q2 = np.percentile(neg_hard, 98)
    q5 = np.percentile(neg_hard, 95)
    q10 = np.percentile(neg_hard, 90)
    
    strong_name = sum(all_negatives['name_levenshtein_sim'] >= 0.70)
    
    # Get Baseline FPs (approx by training on whole dataset or just look at prior run)
    # We will just do the fold loop and compute FPs from baseline preds
    # ----------------------------------------
    
    base_cols = [c for c in feature_df.columns if c not in ["source1_entity_id", "target_entity_id", "target_source", "label", "hardness"]]
    X = feature_df[base_cols].values
    y = feature_df['label'].values
    groups = feature_df['source1_entity_id'].values
    
    gkf = GroupKFold(n_splits=5)
    
    xgb_params = {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8, 'tree_method': 'hist', 'n_jobs': -1, 'random_state': 42}
    lgb_params = {'objective': 'binary', 'metric': 'binary_logloss', 'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'verbose': -1}
    
    base_preds = []
    hn_preds = []
    
    print("Training baseline and hard-negative models...")
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        h_train = feature_df['hardness'].values[train_idx]
        
        # Calculate sample weights for HN model
        # Only use training fold to calculate 90th percentile threshold!
        train_negs = h_train[y_train == 0]
        q10_train = np.percentile(train_negs, 90)
        
        sample_weights = np.ones(len(y_train))
        for i in range(len(y_train)):
            if y_train[i] == 0 and h_train[i] >= q10_train:
                sample_weights[i] = 3.0
                
        # Train Baseline
        m_xgb_b = xgb.XGBClassifier(**xgb_params)
        m_xgb_b.fit(X_train, y_train)
        p_xgb_b = m_xgb_b.predict_proba(X_val)[:, 1]
        
        m_lgb_b = lgb.LGBMClassifier(**lgb_params)
        m_lgb_b.fit(X_train, y_train)
        p_lgb_b = m_lgb_b.predict_proba(X_val)[:, 1]
        
        p_blend_b = 0.75 * p_xgb_b + 0.25 * p_lgb_b
        
        val_df_b = feature_df.iloc[val_idx][['source1_entity_id', 'target_entity_id', 'label', 'hardness']].copy()
        val_df_b['prob'] = p_blend_b
        base_preds.append(val_df_b)
        
        # Train Hard-Negative
        m_xgb_hn = xgb.XGBClassifier(**xgb_params)
        m_xgb_hn.fit(X_train, y_train, sample_weight=sample_weights)
        p_xgb_hn = m_xgb_hn.predict_proba(X_val)[:, 1]
        
        m_lgb_hn = lgb.LGBMClassifier(**lgb_params)
        m_lgb_hn.fit(X_train, y_train, sample_weight=sample_weights)
        p_lgb_hn = m_lgb_hn.predict_proba(X_val)[:, 1]
        
        p_blend_hn = 0.75 * p_xgb_hn + 0.25 * p_lgb_hn
        
        val_df_hn = feature_df.iloc[val_idx][['source1_entity_id', 'target_entity_id', 'label', 'hardness']].copy()
        val_df_hn['prob'] = p_blend_hn
        hn_preds.append(val_df_hn)
        
    df_base = pd.concat(base_preds, ignore_index=True)
    df_hn = pd.concat(hn_preds, ignore_index=True)
    
    df_base = df_base.merge(s1_dev[['entity_id', 'country']], left_on='source1_entity_id', right_on='entity_id')
    df_hn = df_hn.merge(s1_dev[['entity_id', 'country']], left_on='source1_entity_id', right_on='entity_id')
    
    res_base = evaluate_fold_preds(df_base, dev_gt)
    res_hn = evaluate_fold_preds(df_hn, dev_gt)
    
    res_base_in = evaluate_fold_preds(df_base[df_base['country'] == 'India'], dev_gt)
    res_base_us = evaluate_fold_preds(df_base[df_base['country'] == 'US'], dev_gt)
    
    res_hn_in = evaluate_fold_preds(df_hn[df_hn['country'] == 'India'], dev_gt)
    res_hn_us = evaluate_fold_preds(df_hn[df_hn['country'] == 'US'], dev_gt)
    
    # Error Analysis
    base_fps = set(zip(df_base[(df_base['prob']>=0.75) & (df_base['label']==0)]['source1_entity_id'], df_base[(df_base['prob']>=0.75) & (df_base['label']==0)]['target_entity_id']))
    hn_fps = set(zip(df_hn[(df_hn['prob']>=0.75) & (df_hn['label']==0)]['source1_entity_id'], df_hn[(df_hn['prob']>=0.75) & (df_hn['label']==0)]['target_entity_id']))
    
    base_tps = set(zip(df_base[(df_base['prob']>=0.75) & (df_base['label']==1)]['source1_entity_id'], df_base[(df_base['prob']>=0.75) & (df_base['label']==1)]['target_entity_id']))
    hn_tps = set(zip(df_hn[(df_hn['prob']>=0.75) & (df_hn['label']==1)]['source1_entity_id'], df_hn[(df_hn['prob']>=0.75) & (df_hn['label']==1)]['target_entity_id']))
    
    base_fps_disappear = len(base_fps - hn_fps)
    base_tps_lost = len(base_tps - hn_tps)
    base_fns_become_tps = len(hn_tps - base_tps)
    new_fps = len(hn_fps - base_fps)
    
    # Analyze hardness of FP vs TP
    fp_hard = df_base[(df_base['prob']>=0.75) & (df_base['label']==0)]['hardness'].values
    
    print(f"BASELINE: Macro F0.5 = {res_base['Macro_F05']:.4f} ± {res_base['Macro_F05_std']:.4f}")
    print(f"HARD-NEG: Macro F0.5 = {res_hn['Macro_F05']:.4f} ± {res_hn['Macro_F05_std']:.4f}")
    
    results = {
        "hardness_dist": {
            "total_negatives": len(all_negatives),
            "q1": float(q1), "q2": float(q2), "q5": float(q5), "q10": float(q10),
            "strong_name_count": int(strong_name),
            "median_tp_hardness": float(np.median(pos_hard)),
            "median_fp_hardness": float(np.median(fp_hard)),
            "median_neg_hardness": float(np.median(neg_hard))
        },
        "baseline": {
            "macro_f05": res_base['Macro_F05'], "macro_f05_std": res_base['Macro_F05_std'],
            "precision": res_base['Precision'], "recall": res_base['Recall'],
            "tp": res_base['TP'], "fp": res_base['FP'], "fn": res_base['FN'],
            "zm_fp": res_base['ZM_FP_Rate'], "singleton_acc": res_base['Singleton_Acc'], "multi_macro": res_base['Multi_Macro'],
            "in_macro_f05": res_base_in['Macro_F05'], "us_macro_f05": res_base_us['Macro_F05']
        },
        "hard_neg": {
            "macro_f05": res_hn['Macro_F05'], "macro_f05_std": res_hn['Macro_F05_std'],
            "precision": res_hn['Precision'], "recall": res_hn['Recall'],
            "tp": res_hn['TP'], "fp": res_hn['FP'], "fn": res_hn['FN'],
            "zm_fp": res_hn['ZM_FP_Rate'], "singleton_acc": res_hn['Singleton_Acc'], "multi_macro": res_hn['Multi_Macro'],
            "in_macro_f05": res_hn_in['Macro_F05'], "us_macro_f05": res_hn_us['Macro_F05']
        },
        "error_analysis": {
            "base_fps_disappear": base_fps_disappear,
            "base_tps_lost": base_tps_lost,
            "base_fns_become_tps": base_fns_become_tps,
            "new_fps": new_fps
        }
    }
    
    with open('output/phase18_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
