import os
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
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
    
    zm_entities = [s for s in all_s1_ids if entity_stats[s]['gt_len'] == 0]
    zm_fp_rate = sum(1 for s in zm_entities if entity_stats[s]['fp'] > 0) / max(1, len(zm_entities))
    
    sg_entities = [s for s in all_s1_ids if entity_stats[s]['gt_len'] == 1]
    sg_acc = sum(1 for s in sg_entities if entity_stats[s]['tp'] == 1 and entity_stats[s]['fp'] == 0) / max(1, len(sg_entities))
    
    mm_entities = [s for s in all_s1_ids if entity_stats[s]['gt_len'] > 1]
    mm_f05 = np.mean([entity_stats[s]['f05'] for s in mm_entities]) if mm_entities else 0.0
    
    return {
        'Macro_F05': macro_f05,
        'ZM_FP_Rate': zm_fp_rate,
        'Singleton_Acc': sg_acc,
        'MultiMatch_F05': mm_f05,
        'Precision': tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else 0.0,
        'Recall': tp_total / (tp_total + fn_total) if tp_total + fn_total > 0 else 0.0,
        'entity_stats': entity_stats
    }

def run_experiment(feature_df, dev_gt, s1_dev, use_new_features):
    print(f"\n--- Running Experiment: {'NEW_FEATURES' if use_new_features else 'BASELINE'} ---")
    
    labels = []
    for s1, t in zip(feature_df['source1_entity_id'], feature_df['target_entity_id']):
        labels.append(1 if t in dev_gt.get(s1, set()) else 0)
    feature_df['label'] = labels
    
    new_feat_names = [
        "address_exact_street_number_match",
        "address_street_number_numeric_match",
        "address_postal_code_exact_match",
        "address_postal_code_prefix_match",
        "address_numeric_token_jaccard",
        "address_number_consistency"
    ]
    
    if use_new_features:
        feature_cols = [c for c in feature_df.columns if c not in ["source1_entity_id", "target_entity_id", "target_source", "label"]]
    else:
        feature_cols = [c for c in feature_df.columns if c not in ["source1_entity_id", "target_entity_id", "target_source", "label"] + new_feat_names]
        
    X = feature_df[feature_cols].values
    y = feature_df['label'].values
    groups = feature_df['source1_entity_id'].values
    
    gkf = GroupKFold(n_splits=5)
    
    xgb_params = {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8, 'tree_method': 'hist', 'n_jobs': -1, 'random_state': 42}
    lgb_params = {'objective': 'binary', 'metric': 'binary_logloss', 'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'verbose': -1}
    
    fold_preds = []
    feat_importances = np.zeros(len(feature_cols))
    
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        m_xgb = xgb.XGBClassifier(**xgb_params)
        m_xgb.fit(X_train, y_train)
        p_xgb = m_xgb.predict_proba(X_val)[:, 1]
        
        feat_importances += m_xgb.feature_importances_
        
        m_lgb = lgb.LGBMClassifier(**lgb_params)
        m_lgb.fit(X_train, y_train)
        p_lgb = m_lgb.predict_proba(X_val)[:, 1]
        
        p_blend = 0.75 * p_xgb + 0.25 * p_lgb
        
        val_df = feature_df.iloc[val_idx].copy()
        val_df['prob'] = p_blend
        fold_preds.append(val_df)
        
    full_preds = pd.concat(fold_preds, ignore_index=True)
    full_preds = full_preds.merge(s1_dev[['entity_id', 'country']], left_on='source1_entity_id', right_on='entity_id')
    
    thresholds = [0.60, 0.65, 0.70, 0.75, 0.80]
    thresh_results = []
    for thresh in thresholds:
        metrics = evaluate_fold_preds(full_preds, dev_gt, thresh)
        metrics['Threshold'] = thresh
        
        metrics_in = evaluate_fold_preds(full_preds[full_preds['country'] == 'India'], dev_gt, thresh)
        metrics_us = evaluate_fold_preds(full_preds[full_preds['country'] == 'US'], dev_gt, thresh)
        metrics['India_Macro_F05'] = metrics_in['Macro_F05']
        metrics['US_Macro_F05'] = metrics_us['Macro_F05']
        metrics['India_ZM_FP'] = metrics_in['ZM_FP_Rate']
        metrics['US_ZM_FP'] = metrics_us['ZM_FP_Rate']
        
        thresh_results.append(metrics)
        
    res_df = pd.DataFrame([{k: v for k, v in m.items() if k != 'entity_stats'} for m in thresh_results])
    print(res_df[['Threshold', 'Macro_F05', 'Precision', 'Recall', 'ZM_FP_Rate', 'Singleton_Acc']])
    
    feat_importances /= 5
    fi_dict = dict(zip(feature_cols, feat_importances))
    
    return {
        'type': 'NEW_FEATURES' if use_new_features else 'BASELINE',
        'thresh_results': thresh_results,
        'full_preds': full_preds,
        'feature_importance': fi_dict
    }

def main():
    seed = 42
    sample_size = 5000
    target_scan_limit = 100_000

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

    config = BlockingConfig(
        use_stem_prefix=True, use_transliteration=True, use_rare_name_token=True,
        use_addr_alpha=True, max_candidates_per_entity=100, max_key_frequency=5000
    )
    config.channel_weights = {}
    generator = CandidateGenerator(config=config)
    s2_path = get_source_path("train", 2)
    s3_path = get_source_path("train", 3)

    raw_targets = {}
    indexed_s2 = 0
    for r in iter_source_records(s2_path):
        eid = r["entity_id"]
        if eid in target_ids_needed or indexed_s2 < target_scan_limit:
            generator.index_target_record(eid, r.get("business_name", ""), r.get("business_address", ""), r.get("country", ""))
            raw_targets[eid] = {"entity_id": eid, "name": r.get("business_name", ""), "address": r.get("business_address", ""), "country": r.get("country", ""), "source": "S2"}
            indexed_s2 += 1
            
    indexed_s3 = 0
    for r in iter_source_records(s3_path):
        eid = r["entity_id"]
        if eid in target_ids_needed or indexed_s3 < target_scan_limit:
            generator.index_target_record(eid, r.get("business_name", ""), r.get("business_address", ""), r.get("country", ""))
            raw_targets[eid] = {"entity_id": eid, "name": r.get("business_name", ""), "address": r.get("business_address", ""), "country": r.get("country", ""), "source": "S3"}
            indexed_s3 += 1
            
    generator.prune_high_frequency_keys()
    candidates = generator.generate_candidates_for_dataframe(s1_dev)
    
    feat_cfg = FeatureConfig()
    extractor = PairwiseFeatureExtractor(config=feat_cfg)
    s1_preprocessed = {}
    for _, row in s1_dev.iterrows():
        eid = str(row["entity_id"])
        s1_preprocessed[eid] = extractor.preprocess_entity(eid, row.get("business_name", ""), row.get("business_address", ""), row.get("country", ""), "S1")
    candidate_target_ids = {c.matched_entity_id for c in candidates}
    target_preprocessed = {}
    for tid in candidate_target_ids:
        r = raw_targets[tid]
        target_preprocessed[tid] = extractor.preprocess_entity(tid, r["name"], r["address"], r["country"], r["source"])
        
    feature_df = extractor.extract_features_for_candidates(candidates, s1_preprocessed, target_preprocessed)
    
    res_base = run_experiment(feature_df, dev_gt, s1_dev, False)
    res_new = run_experiment(feature_df, dev_gt, s1_dev, True)
    
    with open(ROOT / "output/phase11_results.pkl", "wb") as f:
        pickle.dump({'Baseline': res_base, 'New': res_new}, f)
        
    summary_rows = []
    for r in [res_base, res_new]:
        for tr in r['thresh_results']:
            summary_rows.append({
                'System': r['type'],
                'Threshold': tr['Threshold'],
                'Macro_F05': tr['Macro_F05'],
                'Precision': tr['Precision'],
                'Recall': tr['Recall'],
                'ZM_FP_Rate': tr['ZM_FP_Rate'],
                'Singleton_Acc': tr['Singleton_Acc'],
                'MultiMatch_F05': tr['MultiMatch_F05'],
                'India_Macro_F05': tr['India_Macro_F05'],
                'US_Macro_F05': tr['US_Macro_F05']
            })
            
    pd.DataFrame(summary_rows).to_csv(ROOT / "output/phase_11_results.tsv", sep='\t', index=False)
    
    new_feats = [
        "address_exact_street_number_match",
        "address_street_number_numeric_match",
        "address_postal_code_exact_match",
        "address_postal_code_prefix_match",
        "address_numeric_token_jaccard",
        "address_number_consistency"
    ]
    print("\nFeature Importances (XGBoost) for NEW features:")
    for nf in new_feats:
        print(f"{nf}: {res_new['feature_importance'].get(nf, 0.0):.5f}")

    base_preds = res_base['full_preds']
    new_preds = res_new['full_preds']
    base_m = base_preds[base_preds['prob'] >= 0.75]
    new_m = new_preds[new_preds['prob'] >= 0.75]
    
    b_pairs = set(zip(base_m['source1_entity_id'], base_m['target_entity_id']))
    n_pairs = set(zip(new_m['source1_entity_id'], new_m['target_entity_id']))
    
    recovered = n_pairs - b_pairs
    lost = b_pairs - n_pairs
    
    rec_tp = [p for p in recovered if p[1] in dev_gt.get(p[0], [])]
    rec_fp = [p for p in recovered if p[1] not in dev_gt.get(p[0], [])]
    
    lost_tp = [p for p in lost if p[1] in dev_gt.get(p[0], [])]
    
    print(f"\nModel FN Recovered by new features: {len(rec_tp)}")
    print(f"Model FN Lost (new false negatives): {len(lost_tp)}")
    print(f"New False Positives added: {len(rec_fp)}")

if __name__ == "__main__":
    main()
