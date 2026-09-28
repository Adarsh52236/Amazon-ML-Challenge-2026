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

def evaluate_fold_preds(preds_df, gt_dict, name="Model", rule='baseline'):
    preds_dict = defaultdict(list)
    
    for _, row in preds_df.iterrows():
        s1_id = row['source1_entity_id']
        t_id = row['target_entity_id']
        prob = row['prob']
        max_prob = row['max_prob']
        
        keep = False
        if rule == 'baseline':
            keep = prob >= 0.75
        elif rule == 'rule_a':
            keep = (prob >= 0.75) and (max_prob > 0 and prob / max_prob >= 0.80)
        elif rule == 'rule_b':
            keep = (prob >= 0.75) and (prob >= max_prob - 0.15)
            
        if keep:
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

def analyze_s1_distributions(preds_df, dev_gt):
    s1_stats = []
    
    for s1_id, group in preds_df.groupby('source1_entity_id'):
        gt_len = len(dev_gt.get(s1_id, []))
        if gt_len == 0:
            cat = "zero-match"
        elif gt_len == 1:
            cat = "singleton"
        else:
            cat = "multi-match"
            
        probs = group['prob'].sort_values(ascending=False).values
        num_cands = len(probs)
        max_p = probs[0] if num_cands > 0 else 0.0
        sec_p = probs[1] if num_cands > 1 else 0.0
        mean_p = np.mean(probs) if num_cands > 0 else 0.0
        std_p = np.std(probs) if num_cands > 0 else 0.0
        
        s1_stats.append({
            's1_id': s1_id,
            'category': cat,
            'num_cands': num_cands,
            'max_prob': max_p,
            'second_prob': sec_p,
            'mean_prob': mean_p,
            'std_prob': std_p,
            'gap': max_p - sec_p,
            'gt50': sum(probs > 0.50),
            'gt60': sum(probs > 0.60),
            'gt70': sum(probs > 0.70),
            'gt75': sum(probs > 0.75),
            'gt80': sum(probs > 0.80),
            'gt90': sum(probs > 0.90)
        })
        
    df_stats = pd.DataFrame(s1_stats)
    
    cats = ['zero-match', 'singleton', 'multi-match']
    analysis = {}
    for c in cats:
        sub = df_stats[df_stats['category'] == c]
        analysis[c] = {
            'count': len(sub),
            'avg_cands': float(sub['num_cands'].mean()),
            'avg_max_prob': float(sub['max_prob'].mean()),
            'avg_second_prob': float(sub['second_prob'].mean()),
            'avg_gap': float(sub['gap'].mean()),
            'avg_gt75': float(sub['gt75'].mean()),
        }
        
    # Analyze Specific Pairs
    tp_below_75 = 0
    fp_above_75 = 0
    s1_both_above_75 = 0
    
    for s1_id, group in preds_df.groupby('source1_entity_id'):
        gt_set = set(dev_gt.get(s1_id, []))
        
        has_gt_above = False
        has_fp_above = False
        
        for _, row in group.iterrows():
            t_id = row['target_entity_id']
            p = row['prob']
            is_gt = t_id in gt_set
            
            if is_gt and p < 0.75:
                tp_below_75 += 1
            if not is_gt and p >= 0.75:
                fp_above_75 += 1
                
            if p >= 0.75:
                if is_gt: has_gt_above = True
                else: has_fp_above = True
                
        if has_gt_above and has_fp_above:
            s1_both_above_75 += 1
            
    analysis['tp_below_75'] = tp_below_75
    analysis['fp_above_75'] = fp_above_75
    analysis['s1_both_above_75'] = s1_both_above_75
    
    return analysis

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
    
    base_cols = [c for c in feature_df.columns if c not in ["source1_entity_id", "target_entity_id", "target_source", "label"]]
    
    X_base = feature_df[base_cols].values
    y = feature_df['label'].values
    groups = feature_df['source1_entity_id'].values
    
    gkf = GroupKFold(n_splits=5)
    
    xgb_params = {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8, 'tree_method': 'hist', 'n_jobs': -1, 'random_state': 42}
    lgb_params = {'objective': 'binary', 'metric': 'binary_logloss', 'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'verbose': -1}
    
    base_preds = []
    
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
        
        val_df = feature_df.iloc[val_idx][['source1_entity_id', 'target_entity_id']].copy()
        val_df['prob'] = p_blend
        base_preds.append(val_df)
        
    df_base = pd.concat(base_preds, ignore_index=True)
    df_base = df_base.merge(s1_dev[['entity_id', 'country']], left_on='source1_entity_id', right_on='entity_id')
    
    # Calculate max prob per S1
    max_probs = df_base.groupby('source1_entity_id')['prob'].max().reset_index()
    max_probs.rename(columns={'prob': 'max_prob'}, inplace=True)
    df_base = df_base.merge(max_probs, on='source1_entity_id', how='left')
    
    dist_analysis = analyze_s1_distributions(df_base, dev_gt)
    
    # Evaluate Rules
    res_base = evaluate_fold_preds(df_base, dev_gt, "Baseline", rule='baseline')
    res_a = evaluate_fold_preds(df_base, dev_gt, "Rule A", rule='rule_a')
    res_b = evaluate_fold_preds(df_base, dev_gt, "Rule B", rule='rule_b')
    
    res_base_in = evaluate_fold_preds(df_base[df_base['country'] == 'India'], dev_gt, "Baseline", rule='baseline')
    res_base_us = evaluate_fold_preds(df_base[df_base['country'] == 'US'], dev_gt, "Baseline", rule='baseline')
    
    res_a_in = evaluate_fold_preds(df_base[df_base['country'] == 'India'], dev_gt, "Rule A", rule='rule_a')
    res_a_us = evaluate_fold_preds(df_base[df_base['country'] == 'US'], dev_gt, "Rule A", rule='rule_a')
    
    res_b_in = evaluate_fold_preds(df_base[df_base['country'] == 'India'], dev_gt, "Rule B", rule='rule_b')
    res_b_us = evaluate_fold_preds(df_base[df_base['country'] == 'US'], dev_gt, "Rule B", rule='rule_b')
    
    def get_diff(new_stats, base_stats, cat_len=None):
        entities_changed = 0
        cat_affected = 0
        for s1_id in base_stats:
            if new_stats[s1_id]['f05'] != base_stats[s1_id]['f05']:
                entities_changed += 1
                if cat_len is not None:
                    if cat_len == -1 and base_stats[s1_id]['gt_len'] > 1:
                        cat_affected += 1
                    elif base_stats[s1_id]['gt_len'] == cat_len:
                        cat_affected += 1
        return entities_changed, cat_affected
        
    diff_a = {
        'gt_lost': res_base['TP'] - res_a['TP'],
        'fp_removed': res_base['FP'] - res_a['FP'],
        'entities_changed': get_diff(res_a['entity_stats'], res_base['entity_stats'])[0],
        'zm_affected': get_diff(res_a['entity_stats'], res_base['entity_stats'], 0)[1],
        'sg_affected': get_diff(res_a['entity_stats'], res_base['entity_stats'], 1)[1],
        'mm_affected': get_diff(res_a['entity_stats'], res_base['entity_stats'], -1)[1]
    }
    
    diff_b = {
        'gt_lost': res_base['TP'] - res_b['TP'],
        'fp_removed': res_base['FP'] - res_b['FP'],
        'entities_changed': get_diff(res_b['entity_stats'], res_base['entity_stats'])[0],
        'zm_affected': get_diff(res_b['entity_stats'], res_base['entity_stats'], 0)[1],
        'sg_affected': get_diff(res_b['entity_stats'], res_base['entity_stats'], 1)[1],
        'mm_affected': get_diff(res_b['entity_stats'], res_base['entity_stats'], -1)[1]
    }
    
    print(f"BASELINE: Macro F0.5 = {res_base['Macro_F05']:.4f} ± {res_base['Macro_F05_std']:.4f}")
    print(f"RULE A:   Macro F0.5 = {res_a['Macro_F05']:.4f} ± {res_a['Macro_F05_std']:.4f}")
    print(f"RULE B:   Macro F0.5 = {res_b['Macro_F05']:.4f} ± {res_b['Macro_F05_std']:.4f}")
    
    results = {
        "dist_analysis": dist_analysis,
        "baseline": {
            "macro_f05": res_base['Macro_F05'], "macro_f05_std": res_base['Macro_F05_std'],
            "precision": res_base['Precision'], "recall": res_base['Recall'],
            "tp": res_base['TP'], "fp": res_base['FP'], "fn": res_base['FN'],
            "zm_fp": res_base['ZM_FP_Rate'], "singleton_acc": res_base['Singleton_Acc'], "multi_macro": res_base['Multi_Macro'],
            "in_macro_f05": res_base_in['Macro_F05'], "us_macro_f05": res_base_us['Macro_F05']
        },
        "rule_a": {
            "macro_f05": res_a['Macro_F05'], "macro_f05_std": res_a['Macro_F05_std'],
            "precision": res_a['Precision'], "recall": res_a['Recall'],
            "tp": res_a['TP'], "fp": res_a['FP'], "fn": res_a['FN'],
            "zm_fp": res_a['ZM_FP_Rate'], "singleton_acc": res_a['Singleton_Acc'], "multi_macro": res_a['Multi_Macro'],
            "in_macro_f05": res_a_in['Macro_F05'], "us_macro_f05": res_a_us['Macro_F05']
        },
        "rule_b": {
            "macro_f05": res_b['Macro_F05'], "macro_f05_std": res_b['Macro_F05_std'],
            "precision": res_b['Precision'], "recall": res_b['Recall'],
            "tp": res_b['TP'], "fp": res_b['FP'], "fn": res_b['FN'],
            "zm_fp": res_b['ZM_FP_Rate'], "singleton_acc": res_b['Singleton_Acc'], "multi_macro": res_b['Multi_Macro'],
            "in_macro_f05": res_b_in['Macro_F05'], "us_macro_f05": res_b_us['Macro_F05']
        },
        "diff_a": diff_a,
        "diff_b": diff_b
    }
    
    with open('output/phase17_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
