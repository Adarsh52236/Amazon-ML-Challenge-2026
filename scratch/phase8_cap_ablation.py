import os
import sys
import time
import psutil
from pathlib import Path
import json

import numpy as np
import pandas as pd
from collections import defaultdict
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import scipy.stats as stats

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
    zm_avg_pred = np.mean([entity_stats[s]['fp'] for s in zm_entities]) if zm_entities else 0.0
    
    sg_entities = [s for s in all_s1_ids if entity_stats[s]['gt_len'] == 1]
    sg_acc = sum(1 for s in sg_entities if entity_stats[s]['tp'] == 1 and entity_stats[s]['fp'] == 0) / max(1, len(sg_entities))
    sg_fp_rate = sum(1 for s in sg_entities if entity_stats[s]['fp'] > 0) / max(1, len(sg_entities))
    sg_prec = np.mean([entity_stats[s]['tp'] / (entity_stats[s]['tp'] + entity_stats[s]['fp']) if (entity_stats[s]['tp'] + entity_stats[s]['fp']) > 0 else 0.0 for s in sg_entities]) if sg_entities else 0.0
    sg_rec = np.mean([entity_stats[s]['tp'] for s in sg_entities]) if sg_entities else 0.0
    
    mm_entities = [s for s in all_s1_ids if entity_stats[s]['gt_len'] > 1]
    mm_f05 = np.mean([entity_stats[s]['f05'] for s in mm_entities]) if mm_entities else 0.0
    mm_full = sum(1 for s in mm_entities if entity_stats[s]['fn'] == 0) / max(1, len(mm_entities))
    mm_partial = sum(1 for s in mm_entities if entity_stats[s]['fn'] > 0 and entity_stats[s]['tp'] > 0) / max(1, len(mm_entities))
    mm_zero = sum(1 for s in mm_entities if entity_stats[s]['tp'] == 0) / max(1, len(mm_entities))
    mm_rec = np.mean([entity_stats[s]['tp'] / entity_stats[s]['gt_len'] for s in mm_entities]) if mm_entities else 0.0
    
    return {
        'Macro_F05': macro_f05,
        'ZM_FP_Rate': zm_fp_rate,
        'ZM_Avg_Pred': zm_avg_pred,
        'Singleton_Acc': sg_acc,
        'Singleton_FP_Rate': sg_fp_rate,
        'Singleton_Prec': sg_prec,
        'Singleton_Rec': sg_rec,
        'MultiMatch_F05': mm_f05,
        'MultiMatch_Full': mm_full,
        'MultiMatch_Partial': mm_partial,
        'MultiMatch_Zero': mm_zero,
        'MultiMatch_Rec': mm_rec,
        'Precision': tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else 0.0,
        'Recall': tp_total / (tp_total + fn_total) if tp_total + fn_total > 0 else 0.0,
    }

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def main():
    print("Loading data...")
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

    # Configuration for Uncapped
    config = BlockingConfig(
        use_stem_prefix=True, use_transliteration=True, use_rare_name_token=True,
        use_addr_alpha=True, max_candidates_per_entity=100000, max_key_frequency=5000
    )
    # Simulate buggy behavior: length of channels
    config.channel_weights = {}

    generator = CandidateGenerator(config=config)
    s2_path = get_source_path("train", 2)
    s3_path = get_source_path("train", 3)

    raw_targets = {}
    indexed_s2 = 0
    t0 = time.time()
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
    
    gen_time_total = time.time() - t0
    print(f"Index built in {gen_time_total:.2f}s")
    
    print("Generating Uncapped Candidates...")
    t0 = time.time()
    cands_uncapped = generator.generate_candidates_for_dataframe(s1_dev)
    gen_time_total += time.time() - t0
    
    # Sort ALL candidates manually by (len(channels), len(channels))
    # This precisely mimics the CandidateGenerator stable internal sort logic when channel_weights is empty
    def score_c(c):
        return (len(c.channels), len(c.channels))
        
    s1_to_cands = defaultdict(list)
    for c in cands_uncapped:
        s1_to_cands[c.source1_entity_id].append(c)
        
    for s1_id in s1_to_cands:
        s1_to_cands[s1_id].sort(key=score_c, reverse=True)
        
    # Pre-extract ALL features
    # Since we need Uncapped, we must extract features for all 5.7M candidates.
    print(f"Extracting Features for {len(cands_uncapped)} Uncapped Candidates...")
    feat_cfg = FeatureConfig()
    extractor = PairwiseFeatureExtractor(config=feat_cfg)
    s1_preprocessed = {}
    for _, row in s1_dev.iterrows():
        eid = str(row["entity_id"])
        s1_preprocessed[eid] = extractor.preprocess_entity(eid, row.get("business_name", ""), row.get("business_address", ""), row.get("country", ""), "S1")
        
    candidate_target_ids = {c.matched_entity_id for c in cands_uncapped}
    target_preprocessed = {}
    for tid in candidate_target_ids:
        r = raw_targets[tid]
        target_preprocessed[tid] = extractor.preprocess_entity(tid, r["name"], r["address"], r["country"], r["source"])
        
    t0 = time.time()
    feature_df_uncapped = extractor.extract_features_for_candidates(cands_uncapped, s1_preprocessed, target_preprocessed)
    ext_time_total = time.time() - t0
    print(f"Features extracted in {ext_time_total:.2f}s")
    
    labels = []
    for s1, t in zip(feature_df_uncapped['source1_entity_id'], feature_df_uncapped['target_entity_id']):
        labels.append(1 if t in dev_gt.get(s1, set()) else 0)
    feature_df_uncapped['label'] = labels
    
    # For fast filtering
    # Since cands_uncapped is NOT sorted inside feature_df_uncapped (it's in original insertion order before our sort),
    # Wait, extractor.extract_features_for_candidates preserves the order of candidates list passed to it!
    # Let's pass the sorted candidates!
    
    sorted_all_cands = []
    for s1_id in s1_dev['entity_id']:
        sorted_all_cands.extend(s1_to_cands[s1_id])
        
    print("Extracting features (Sorted order)...")
    feature_df_uncapped = extractor.extract_features_for_candidates(sorted_all_cands, s1_preprocessed, target_preprocessed)
    
    labels = []
    for s1, t in zip(feature_df_uncapped['source1_entity_id'], feature_df_uncapped['target_entity_id']):
        labels.append(1 if t in dev_gt.get(s1, set()) else 0)
    feature_df_uncapped['label'] = labels
    feature_df_uncapped['rank'] = feature_df_uncapped.groupby('source1_entity_id').cumcount() + 1
    
    caps_to_test = [25, 50, 75, 100, 125, 150, 200, 250, 300, 500, 100000]
    
    feature_cols = [c for c in feature_df_uncapped.columns if c not in ["source1_entity_id", "target_entity_id", "target_source", "label", "rank"]]
    
    xgb_params = {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8, 'tree_method': 'hist', 'n_jobs': -1, 'random_state': 42}
    lgb_params = {'objective': 'binary', 'metric': 'binary_logloss', 'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'verbose': -1}
    
    gkf = GroupKFold(n_splits=5)
    
    cap_results = []
    fold_macro_results = []
    
    for cap in caps_to_test:
        is_uncapped = (cap == 100000)
        cap_str = 'Uncapped' if is_uncapped else cap
        print(f"\n--- Evaluating Cap: {cap_str} ---")
        
        mem_start = get_memory_mb()
        
        t0 = time.time()
        cap_df = feature_df_uncapped[feature_df_uncapped['rank'] <= cap].copy().reset_index(drop=True)
        gen_time = gen_time_total # In a real system, smaller cap = faster sort, but we just use total here
        ext_time = ext_time_total * (len(cap_df) / len(feature_df_uncapped)) # Approximation of time if we only extracted cap
        
        # Calculate Candidate Metrics
        cands_per_s1 = cap_df.groupby('source1_entity_id').size()
        total_cands = len(cap_df)
        gt_found = cap_df['label'].sum()
        cand_recall = gt_found / len(all_gt_pairs)
        
        india_s1 = set(s1_dev[s1_dev['country'] == 'India']['entity_id'])
        us_s1 = set(s1_dev[s1_dev['country'] == 'US']['entity_id'])
        
        india_gt_total = sum(1 for s1, t in all_gt_pairs if s1 in india_s1)
        us_gt_total = sum(1 for s1, t in all_gt_pairs if s1 in us_s1)
        india_gt_found = cap_df[cap_df['source1_entity_id'].isin(india_s1)]['label'].sum()
        us_gt_found = cap_df[cap_df['source1_entity_id'].isin(us_s1)]['label'].sum()
        
        india_recall = india_gt_found / max(1, india_gt_total)
        us_recall = us_gt_found / max(1, us_gt_total)
        
        # Entities losing all GT
        all_gt_s1 = {s for s, t in all_gt_pairs}
        cap_gt_s1 = set(cap_df[cap_df['label'] == 1]['source1_entity_id'])
        entities_all_lost = len(all_gt_s1 - cap_gt_s1)
        
        india_lost_all = len((all_gt_s1 & india_s1) - cap_gt_s1)
        us_lost_all = len((all_gt_s1 & us_s1) - cap_gt_s1)
        
        X = cap_df[feature_cols].values
        y = cap_df['label'].values
        groups = cap_df['source1_entity_id'].values
        
        fold_preds = []
        fold_idx = 1
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
            
            val_df = cap_df.iloc[val_idx].copy()
            val_df['prob'] = p_blend
            val_df['fold'] = fold_idx
            fold_preds.append(val_df)
            fold_idx += 1
            
        full_preds = pd.concat(fold_preds, ignore_index=True)
        full_preds = full_preds.merge(s1_dev[['entity_id', 'country']], left_on='source1_entity_id', right_on='entity_id')
        
        ml_time = time.time() - t0
        mem_peak = get_memory_mb()
        
        thresholds = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        for thresh in thresholds:
            metrics = evaluate_fold_preds(full_preds, dev_gt, thresh)
            
            in_metrics = evaluate_fold_preds(full_preds[full_preds['country'] == 'India'], dev_gt, thresh)
            us_metrics = evaluate_fold_preds(full_preds[full_preds['country'] == 'US'], dev_gt, thresh)
            
            res = {
                'Cap': cap_str,
                'Threshold': thresh,
                'Cand_Total': total_cands,
                'Cand_Mean': cands_per_s1.mean(),
                'Cand_Median': cands_per_s1.median(),
                'Cand_P95': cands_per_s1.quantile(0.95),
                'Cand_P99': cands_per_s1.quantile(0.99),
                'Cand_Max': cands_per_s1.max(),
                'GT_Found': gt_found,
                'Cand_Recall': cand_recall,
                'GT_Lost': len(all_gt_pairs) - gt_found,
                'Entities_All_Lost': entities_all_lost,
                'India_Recall': india_recall,
                'US_Recall': us_recall,
                'India_GT_Lost': india_gt_total - india_gt_found,
                'US_GT_Lost': us_gt_total - us_gt_found,
                'India_Lost_All': india_lost_all,
                'US_Lost_All': us_lost_all,
                'Gen_Time': gen_time,
                'Ext_Time': ext_time,
                'ML_Time': ml_time,
                'Peak_Mem_MB': mem_peak,
                
                'Macro_F05_Mean': metrics['Macro_F05'],
                'Precision': metrics['Precision'],
                'Model_Recall': metrics['Recall'],
                'ZM_Entities': sum(1 for s in dev_s1_ids if len(dev_gt.get(s, [])) == 0),
                'ZM_FP_Rate': metrics['ZM_FP_Rate'],
                'ZM_Avg_Pred': metrics['ZM_Avg_Pred'],
                
                'Singles_Acc': metrics['Singleton_Acc'],
                'Singles_FP': metrics['Singleton_FP_Rate'],
                'Singles_Prec': metrics['Singleton_Prec'],
                'Singles_Rec': metrics['Singleton_Rec'],
                
                'Multi_F05': metrics['MultiMatch_F05'],
                'Multi_Full': metrics['MultiMatch_Full'],
                'Multi_Partial': metrics['MultiMatch_Partial'],
                'Multi_Zero': metrics['MultiMatch_Zero'],
                'Multi_Rec': metrics['MultiMatch_Rec'],
                
                'India_Macro_F05': in_metrics['Macro_F05'],
                'India_ZM_FP': in_metrics['ZM_FP_Rate'],
                'US_Macro_F05': us_metrics['Macro_F05'],
                'US_ZM_FP': us_metrics['ZM_FP_Rate']
            }
            cap_results.append(res)
            
            # Save fold-level Macro F0.5
            for f in range(1, 6):
                f_df = full_preds[full_preds['fold'] == f]
                f_met = evaluate_fold_preds(f_df, dev_gt, thresh)
                fold_macro_results.append({
                    'Cap': cap_str,
                    'Threshold': thresh,
                    'Fold': f,
                    'Macro_F05': f_met['Macro_F05']
                })
                
    pd.DataFrame(cap_results).to_csv(ROOT / "output/phase_8_correct_cap_ablation.tsv", sep='\t', index=False)
    pd.DataFrame(fold_macro_results).to_csv(ROOT / "output/phase_8_fold_level_results.tsv", sep='\t', index=False)
    print("Saved to output/phase_8_correct_cap_ablation.tsv and output/phase_8_fold_level_results.tsv")

if __name__ == "__main__":
    main()
