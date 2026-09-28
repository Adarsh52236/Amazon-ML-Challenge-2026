import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import json

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from business_entity_resolution.preprocessing_config import BlockingConfig
from business_entity_resolution.candidate_generator import CandidateGenerator
from business_entity_resolution.data_loader import load_ground_truth, load_source_file, iter_source_records, get_source_path

def get_weight(channels, config):
    w = config.channel_weights
    return sum(w.get(ch, 1) for ch in channels)

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
    
    return {
        'Macro_F05': macro_f05,
        'Global_F1': g_f1,
        'ZM_FP_Rate': zm_fp_rate,
        'Singleton_Acc': sg_acc,
        'MultiMatch_F05': mm_f05,
    }

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

    # Generate Uncapped Candidates
    config = BlockingConfig(
        use_stem_prefix=True, use_transliteration=True, use_rare_name_token=True,
        use_addr_alpha=True, max_candidates_per_entity=10000, max_key_frequency=5000
    )
    generator = CandidateGenerator(config=config)
    s2_path = get_source_path("train", 2)
    s3_path = get_source_path("train", 3)
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
    
    raw_dict = defaultdict(list)
    for c in candidates:
        raw_dict[c.source1_entity_id].append(c)

    # PART B: Cap Ablation for Candidate Recall
    cap_results = []
    caps_to_test = [100, 150, 200, 300, 500, 10000] # 10000 is uncapped
    for cap in caps_to_test:
        gt_found = 0
        retained = 0
        entities_affected = 0
        entities_all_lost = 0
        country_recall = defaultdict(lambda: [0, 0])
        cand_counts = []
        for s1_id in dev_s1_ids:
            gt_set = set(dev_gt.get(s1_id, []))
            ctry = str(s1_dev[s1_dev['entity_id'] == s1_id]['country'].values[0]).strip()
            country_recall[ctry][1] += len(gt_set)
            
            cands = raw_dict.get(s1_id, [])
            capped_cands = cands[:cap]
            retained += len(capped_cands)
            cand_counts.append(len(capped_cands))
            
            capped_tids = set(c.matched_entity_id for c in capped_cands)
            capped_gt = gt_set & capped_tids
            gt_found += len(capped_gt)
            country_recall[ctry][0] += len(capped_gt)
            
            raw_tids = set(c.matched_entity_id for c in cands)
            if len(gt_set & raw_tids) > len(capped_gt):
                entities_affected += 1
                if len(capped_gt) == 0:
                    entities_all_lost += 1
                    
        cap_results.append({
            'Cap': 'Uncapped' if cap == 10000 else cap,
            'Retained': retained,
            'Recall': gt_found / len(all_gt_pairs),
            'Entities_Affected': entities_affected,
            'Entities_All_Lost': entities_all_lost,
            'Avg_Cands': np.mean(cand_counts),
            'P95_Cands': np.percentile(cand_counts, 95),
            'P99_Cands': np.percentile(cand_counts, 99),
            'India_Recall': country_recall['India'][0] / max(1, country_recall['India'][1]),
            'US_Recall': country_recall['US'][0] / max(1, country_recall['US'][1])
        })
        
    pd.DataFrame(cap_results).to_csv(ROOT / "output/cap_ablation_recall.csv", index=False)

    # PART E: Deterministic Tie-Break
    det_retained = 0
    det_entities_all_lost = 0
    for s1_id in dev_s1_ids:
        gt_set = set(dev_gt.get(s1_id, []))
        cands = raw_dict.get(s1_id, [])
        # Deterministic sort
        det_cands = sorted(cands, key=lambda c: (get_weight(c.channels, config), len(c.channels), c.matched_entity_id), reverse=True)
        capped_tids = set(c.matched_entity_id for c in det_cands[:100])
        det_retained += len(gt_set & capped_tids)
        if len(gt_set & set(c.matched_entity_id for c in cands)) > 0 and len(gt_set & capped_tids) == 0:
            det_entities_all_lost += 1
            
    print(f"Deterministic Tie-Break (Cap 100): Retained {det_retained} GT pairs, {det_entities_all_lost} entities lost all.")

    # PART F & G: Ranking Quality & India Deep Dive
    rank_dist_us = defaultdict(int)
    rank_dist_in = defaultdict(int)
    for s1_id in dev_s1_ids:
        cands = raw_dict.get(s1_id, [])
        if len(cands) <= 100:
            continue
        ctry = str(s1_dev[s1_dev['entity_id'] == s1_id]['country'].values[0]).strip()
        gt_set = set(dev_gt.get(s1_id, []))
        for i, c in enumerate(cands):
            if c.matched_entity_id in gt_set:
                rank = i + 1
                dist = rank_dist_us if ctry == 'US' else rank_dist_in
                if rank <= 10: dist['1-10'] += 1
                elif rank <= 25: dist['11-25'] += 1
                elif rank <= 50: dist['26-50'] += 1
                elif rank <= 75: dist['51-75'] += 1
                elif rank <= 100: dist['76-100'] += 1
                elif rank <= 150: dist['101-150'] += 1
                elif rank <= 200: dist['151-200'] += 1
                elif rank <= 300: dist['201-300'] += 1
                elif rank <= 500: dist['301-500'] += 1
                else: dist['500+'] += 1
                
    with open(ROOT / "output/ranking_dist.json", "w") as f:
        json.dump({'US': rank_dist_us, 'India': rank_dist_in, 'Det_Retained': det_retained, 'Det_Entities_Lost': det_entities_all_lost}, f, indent=2)

    # PART C: Model Evaluation (for Cap 100, 150, 200, 250)
    feat_path = ROOT / "data/processed/features_dev_phase5_improved.parquet"
    df = pd.read_parquet(feat_path)
    
    # We must simulate caps on the df. Since df doesn't have priority scores directly, 
    # but the features correspond exactly to the first 250 candidates returned by CandidateGenerator!
    # Because CandidateGenerator output is sorted, the first N rows for an S1 are the capped ones.
    df['rank'] = df.groupby('source1_entity_id').cumcount() + 1
    
    labels = []
    for s1, t in zip(df['source1_entity_id'], df['target_entity_id']):
        labels.append(1 if t in dev_gt.get(s1, set()) else 0)
    df['label'] = labels
    
    feature_cols = [c for c in df.columns if c not in ["source1_entity_id", "target_entity_id", "target_source", "label", "rank"]]
    
    gkf = GroupKFold(n_splits=5)
    model_results = []
    
    xgb_params = {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8, 'tree_method': 'hist', 'n_jobs': -1, 'random_state': 42}
    lgb_params = {'objective': 'binary', 'metric': 'binary_logloss', 'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'verbose': -1}
    
    for cap_limit in [100, 150, 200, 250]:
        print(f"Evaluating Model for Cap = {cap_limit}")
        cap_df = df[df['rank'] <= cap_limit].copy().reset_index(drop=True)
        X = cap_df[feature_cols].values
        y = cap_df['label'].values
        groups = cap_df['source1_entity_id'].values
        
        fold = 1
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
            for thresh in [0.55, 0.60, 0.65, 0.70, 0.75]:
                for model_name, probs in [('XGB', p_xgb), ('Blend', p_blend)]:
                    val_df['prob'] = probs
                    metrics = evaluate_fold_preds(val_df, dev_gt, thresh)
                    metrics.update({'Cap': cap_limit, 'Fold': fold, 'Model': model_name, 'Threshold': thresh})
                    model_results.append(metrics)
            fold += 1
            
    pd.DataFrame(model_results).to_csv(ROOT / "output/cap_ablation_model.csv", index=False)

if __name__ == "__main__":
    main()
