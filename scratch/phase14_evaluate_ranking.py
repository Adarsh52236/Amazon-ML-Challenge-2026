import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
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
            f05 = 1.0 if len(gt_set) == 0 else 0.0
        else:
            prec = tp / (tp + fp) if tp + fp > 0 else 0.0
            rec = tp / (tp + fn) if tp + fn > 0 else 0.0
            f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if prec > 0 or rec > 0 else 0.0
                
        f_scores.append(f05)
        entity_stats[s1_id] = {'tp': tp, 'fp': fp, 'fn': fn, 'f05': f05, 'gt_len': len(gt_set)}
        
    macro_f05 = np.mean(f_scores)
    
    zm_entities = [s for s in all_s1_ids if entity_stats[s]['gt_len'] == 0]
    zm_fp_rate = sum(1 for s in zm_entities if entity_stats[s]['fp'] > 0) / max(1, len(zm_entities))
    
    sg_entities = [s for s in all_s1_ids if entity_stats[s]['gt_len'] == 1]
    sg_acc = sum(1 for s in sg_entities if entity_stats[s]['tp'] == 1 and entity_stats[s]['fp'] == 0) / max(1, len(sg_entities))
    
    return {
        'Macro_F05': macro_f05,
        'Macro_F05_std': np.std(f_scores),
        'ZM_FP_Rate': zm_fp_rate,
        'Singleton_Acc': sg_acc,
        'Precision': tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else 0.0,
        'Recall': tp_total / (tp_total + fn_total) if tp_total + fn_total > 0 else 0.0
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

    dev_gt = {s1_id: gt_mapping.get(s1_id, []) for s1_id in dev_s1_ids if s1_id in gt_mapping}
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

    print("Generating newly ranked candidates...")
    config = BlockingConfig(
        use_stem_prefix=True, use_transliteration=True, use_rare_name_token=True,
        use_addr_alpha=True, max_candidates_per_entity=100, max_key_frequency=5000
    )
    generator = CandidateGenerator(config=config)
    
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
    
    candidates = generator.generate_candidates_for_dataframe(s1_dev)
    
    cap_gt_found = sum(1 for c in candidates if c.matched_entity_id in dev_gt.get(c.source1_entity_id, []))
    cand_recall = cap_gt_found / len(all_gt_pairs)
    
    s1_cap_dict = defaultdict(set)
    for c in candidates:
        s1_cap_dict[c.source1_entity_id].add(c.matched_entity_id)
            
    entities_all_lost = sum(1 for s1_id in dev_gt if dev_gt[s1_id] and len(set(dev_gt[s1_id]) & s1_cap_dict[s1_id]) == 0)
    
    s1_country = dict(zip(s1_dev['entity_id'], s1_dev['country']))
    in_cap_gt, us_cap_gt = 0, 0
    in_gt_tot, us_gt_tot = 0, 0
    for s1_id, t_id in all_gt_pairs:
        ctry = s1_country.get(s1_id)
        if ctry == 'India': in_gt_tot += 1
        elif ctry == 'US': us_gt_tot += 1
        if t_id in s1_cap_dict[s1_id]:
            if ctry == 'India': in_cap_gt += 1
            elif ctry == 'US': us_cap_gt += 1
            
    # Original baseline data to compare what was recovered
    phase12_df = pd.read_csv('output/phase_12_ranking_failure_forensics.tsv', sep='\t')
    orig_557_pairs = set(zip(phase12_df['source1_entity_id'], phase12_df['target_entity_id']))
    
    recovered = 0
    for s1, t in orig_557_pairs:
        if t in s1_cap_dict[s1]:
            recovered += 1
            
    # We also need to know newly lost. For that, we compare with old baseline logic.
    # To be exactly precise, we can run old baseline in memory.
    config_uncapped = BlockingConfig(
        use_stem_prefix=True, use_transliteration=True, use_rare_name_token=True,
        use_addr_alpha=True, max_candidates_per_entity=10000, max_key_frequency=5000
    )
    generator_old = CandidateGenerator(config=config_uncapped)
    generator_old.index = generator.index # reuse index!
    old_uncapped_cands = generator_old.generate_candidates_for_dataframe(s1_dev)
    s1_cands_ordered = defaultdict(list)
    for c in old_uncapped_cands:
        s1_cands_ordered[c.source1_entity_id].append(c)
    
    old_cap_dict = defaultdict(set)
    for s1_id in s1_dev['entity_id']:
        s1_id = str(s1_id)
        cands = s1_cands_ordered.get(s1_id, [])
        cands.sort(key=lambda c: len(c.channels), reverse=True)
        capped = cands[:100]
        for c in capped:
            old_cap_dict[c.source1_entity_id].add(c.matched_entity_id)
            
    old_cap_gt_pairs = set()
    new_cap_gt_pairs = set()
    for s1_id, t_id in all_gt_pairs:
        if t_id in old_cap_dict[s1_id]:
            old_cap_gt_pairs.add((s1_id, t_id))
        if t_id in s1_cap_dict[s1_id]:
            new_cap_gt_pairs.add((s1_id, t_id))
            
    new_gt_lost = len(old_cap_gt_pairs - new_cap_gt_pairs)
    net_recovery = len(new_cap_gt_pairs) - len(old_cap_gt_pairs)

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
        
    feature_df = extractor.extract_features_for_candidates(candidates, s1_preprocessed, target_preprocessed)
    
    labels = []
    for s1, t in zip(feature_df['source1_entity_id'], feature_df['target_entity_id']):
        labels.append(1 if t in dev_gt.get(s1, set()) else 0)
    feature_df['label'] = labels
    feature_cols = [c for c in feature_df.columns if c not in ["source1_entity_id", "target_entity_id", "target_source", "label"]]
    
    X = feature_df[feature_cols].values
    y = feature_df['label'].values
    groups = feature_df['source1_entity_id'].values
    
    gkf = GroupKFold(n_splits=5)
    
    xgb_params = {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8, 'tree_method': 'hist', 'n_jobs': -1, 'random_state': 42}
    lgb_params = {'objective': 'binary', 'metric': 'binary_logloss', 'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'verbose': -1}
    
    fold_preds = []
    print("Training models...")
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, X_val = X[train_idx], X[val_idx]
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
        fold_preds.append(val_df)
        
    full_preds = pd.concat(fold_preds, ignore_index=True)
    full_preds = full_preds.merge(s1_dev[['entity_id', 'country']], left_on='source1_entity_id', right_on='entity_id')
    
    metrics = evaluate_fold_preds(full_preds, dev_gt, 0.75)
    
    res = {
        'cand_recall': cand_recall,
        'macro_f05': metrics['Macro_F05'],
        'macro_f05_std': metrics['Macro_F05_std'],
        'precision': metrics['Precision'],
        'recall': metrics['Recall'],
        'zm_fp': metrics['ZM_FP_Rate'],
        'singleton_acc': metrics['Singleton_Acc'],
        'in_cand_recall': in_cap_gt / max(1, in_gt_tot),
        'us_cand_recall': us_cap_gt / max(1, us_gt_tot),
        'recovered': recovered,
        'new_gt_lost': new_gt_lost,
        'net_recovery': net_recovery,
        'entities_lost': entities_all_lost
    }
    
    for k, v in res.items():
        print(f"{k}: {v}")
        
    import json
    with open("output/phase_14_results.json", "w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
