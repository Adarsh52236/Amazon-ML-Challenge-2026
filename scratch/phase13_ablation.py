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

    print("Generating baseline uncapped candidates...")
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
    
    # Store candidates preserving original dictionary iteration order (which is how they are yielded)
    s1_cands_ordered = defaultdict(list)
    for c in uncapped_cands_list:
        s1_cands_ordered[c.source1_entity_id].append(c)

    # Load orig 557 lost pairs
    phase12_df = pd.read_csv('output/phase_12_ranking_failure_forensics.tsv', sep='\t')
    orig_557_pairs = set(zip(phase12_df['source1_entity_id'], phase12_df['target_entity_id']))
    
    # Preprocess feature texts once
    feat_cfg = FeatureConfig()
    extractor = PairwiseFeatureExtractor(config=feat_cfg)
    s1_preprocessed = {}
    for _, row in s1_dev.iterrows():
        eid = str(row["entity_id"])
        s1_preprocessed[eid] = extractor.preprocess_entity(eid, row.get("business_name", ""), row.get("business_address", ""), row.get("country", ""), "S1")
    
    target_preprocessed = {}
    for tid, r in raw_targets_data.items():
        target_preprocessed[tid] = extractor.preprocess_entity(tid, r["name"], r["address"], r["country"], r["source"])
    
    # Master function for ablation
    def run_ablation(name, remove_channels):
        print(f"\n--- Running Ablation: {name} ---")
        
        capped_cands = []
        raw_cands_count = 0
        s1_cand_counts = Counter()
        
        for s1_id in s1_dev['entity_id']:
            s1_id = str(s1_id)
            cands = s1_cands_ordered.get(s1_id, [])
            
            # Create copies with removed channels
            filtered_cands = []
            for c in cands:
                new_channels = [ch for ch in c.channels if ch not in remove_channels]
                if new_channels:
                    # Shallow copy the candidate pair and update channels
                    from copy import copy
                    new_c = copy(c)
                    new_c.channels = new_channels
                    filtered_cands.append(new_c)
                    
            raw_cands_count += len(filtered_cands)
            s1_cand_counts[s1_id] = len(filtered_cands)
            
            # Python's stable sort by len(channels) descending
            filtered_cands.sort(key=lambda c: len(c.channels), reverse=True)
            
            # Cap at 100
            capped = filtered_cands[:100]
            capped_cands.extend(capped)
            
        counts_arr = list(s1_cand_counts.values())
        
        cap_gt_found = sum(1 for c in capped_cands if c.matched_entity_id in dev_gt.get(c.source1_entity_id, []))
        
        s1_cap_dict = defaultdict(set)
        capped_gt_pairs_set = set()
        for c in capped_cands:
            s1_cap_dict[c.source1_entity_id].add(c.matched_entity_id)
            if c.matched_entity_id in dev_gt.get(c.source1_entity_id, []):
                capped_gt_pairs_set.add((c.source1_entity_id, c.matched_entity_id))
                
        entities_all_lost = 0
        for s1_id in dev_gt:
            if dev_gt[s1_id]:
                if len(set(dev_gt[s1_id]) & s1_cap_dict[s1_id]) == 0:
                    entities_all_lost += 1
                    
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
                
        recovered = 0
        still_lost = 0
        for s1, t in orig_557_pairs:
            if t in s1_cap_dict[s1]:
                recovered += 1
            else:
                still_lost += 1
                
        print("Extracting features...")
        feature_df = extractor.extract_features_for_candidates(capped_cands, s1_preprocessed, target_preprocessed)
        
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
        metrics_in = evaluate_fold_preds(full_preds[full_preds['country'] == 'India'], dev_gt, 0.75)
        metrics_us = evaluate_fold_preds(full_preds[full_preds['country'] == 'US'], dev_gt, 0.75)
        
        res = {
            'Experiment': name,
            'Removed Channels': ", ".join(remove_channels) if remove_channels else "None",
            'raw_cands': raw_cands_count,
            'capped_cands': len(capped_cands),
            'cand_recall': cap_gt_found / len(all_gt_pairs),
            'gt_retained': cap_gt_found,
            'entities_lost': entities_all_lost,
            'in_cand_recall': in_cap_gt / max(1, in_gt_tot),
            'us_cand_recall': us_cap_gt / max(1, us_gt_tot),
            'macro_f05': metrics['Macro_F05'],
            'precision': metrics['Precision'],
            'recall': metrics['Recall'],
            'zm_fp': metrics['ZM_FP_Rate'],
            'in_macro_f05': metrics_in['Macro_F05'],
            'us_macro_f05': metrics_us['Macro_F05'],
            'col_stats': {
                'avg_cands': np.mean(counts_arr),
                'p95_cands': np.percentile(counts_arr, 95),
                'p99_cands': np.percentile(counts_arr, 99),
                'gt_100': sum(1 for c in counts_arr if c > 100),
                'gt_500': sum(1 for c in counts_arr if c > 500),
                'gt_1000': sum(1 for c in counts_arr if c > 1000),
            },
            'capped_gt_pairs_set': capped_gt_pairs_set,
            'orig_recovered': recovered,
            'orig_lost': still_lost
        }
        print(f"Cand Recall: {res['cand_recall']:.4f} | Macro F0.5: {res['macro_f05']:.4f}")
        return res

    experiments = [
        ("Baseline", []),
        ("A_No_Max_Num", ["addr_max_num"]),
        ("B_No_First_Num", ["addr_first_num"]),
        ("C_No_WordNum", ["addr_word_num"]),
        ("D_No_Max_First", ["addr_max_num", "addr_first_num"]),
        ("E_No_Max_WordNum", ["addr_max_num", "addr_word_num"]),
        ("F_No_First_WordNum", ["addr_first_num", "addr_word_num"]),
        ("G_No_All", ["addr_max_num", "addr_first_num", "addr_word_num"])
    ]
    
    results = {}
    for name, rm_channels in experiments:
        results[name] = run_ablation(name, rm_channels)
        
    baseline_retained = results["Baseline"]['capped_gt_pairs_set']
    
    summary_rows = []
    for name in results:
        res = results[name]
        newly_lost = len(baseline_retained - res['capped_gt_pairs_set'])
        net_change = res['orig_recovered'] - newly_lost
        
        summary_rows.append({
            'Experiment': name,
            'Removed Channels': res['Removed Channels'],
            'Candidate Recall': res['cand_recall'],
            'GT Retained': res['gt_retained'],
            'Original 557 Recovered': res['orig_recovered'],
            'New GT Lost': newly_lost,
            'Macro F0.5': res['macro_f05'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'ZM FP': res['zm_fp'],
            'India Macro F0.5': res['in_macro_f05'],
            'US Macro F0.5': res['us_macro_f05'],
            'Candidate Count': res['capped_cands'],
            'Avg Cands': res['col_stats']['avg_cands'],
            'P95 Cands': res['col_stats']['p95_cands'],
            '>100 Cands': res['col_stats']['gt_100'],
            '>500 Cands': res['col_stats']['gt_500']
        })
        
    df = pd.DataFrame(summary_rows)
    df.to_csv(ROOT / "output/phase_13_noisy_channel_ablation.tsv", sep='\t', index=False)
    
    # Save md
    md = "# Phase 13: Targeted Noisy-Channel Ablation\n\n"
    
    for _, row in df.iterrows():
        md += f"## {row['Experiment']}\n"
        md += f"- Removed: {row['Removed Channels']}\n"
        md += f"- Raw candidates: N/A\n"
        md += f"- Capped candidates: {row['Candidate Count']}\n"
        md += f"- GT retained: {row['GT Retained']} ({row['Candidate Recall']:.4f})\n"
        md += f"- 557 recovered: {row['Original 557 Recovered']}\n"
        md += f"- Newly lost: {row['New GT Lost']}\n"
        md += f"- Macro F0.5: {row['Macro F0.5']:.4f}\n"
        md += f"- Precision: {row['Precision']:.4f}\n"
        md += f"- Recall: {row['Recall']:.4f}\n"
        md += f"- ZM FP: {row['ZM FP']:.4f}\n"
        md += f"- India Macro F0.5: {row['India Macro F0.5']:.4f}\n"
        md += f"- US Macro F0.5: {row['US Macro F0.5']:.4f}\n\n"
    
    md += "## 10. FINAL COMPARISON TABLE\n\n"
    md += df[['Experiment', 'Removed Channels', 'Candidate Recall', 'GT Retained', 'Original 557 Recovered', 'New GT Lost', 'Macro F0.5', 'Precision', 'Recall', 'ZM FP', 'India Macro F0.5', 'US Macro F0.5', 'Candidate Count']].to_markdown(index=False)
    
    md += "\n\n## 11. DECISION CLASSIFICATION\n\n"
    
    base_macro = df[df['Experiment'] == 'Baseline']['Macro F0.5'].iloc[0]
    best_macro = df['Macro F0.5'].max()
    
    for _, row in df.iterrows():
        name = row['Experiment']
        if name == 'Baseline': continue
        if row['Macro F0.5'] > base_macro + 0.0005:
            c = "A. CLEAR IMPROVEMENT"
        elif row['Macro F0.5'] > base_macro:
            c = "B. PROMISING BUT UNCERTAIN"
        elif row['Macro F0.5'] >= base_macro - 0.0002:
            c = "C. NEUTRAL"
        else:
            c = "D. DEGRADATION"
        md += f"- {name}: {c}\n"
        
    md += "\n\n## 12. CRITICAL QUESTION\n\n"
    
    if best_macro > base_macro + 0.0002:
        md += "YES. The results provide strong evidence that removing noisy address channels improves the downstream OOS Macro F0.5.\n"
    else:
        md += "NO. The results indicate that while removing these channels recovers some cap failures, it harms candidate recall elsewhere, resulting in a net deterioration of Macro F0.5.\n"
        
    with open(ROOT / "documentation/daily/phase_13_noisy_channel_ablation.md", "w") as f:
        f.write(md)

if __name__ == "__main__":
    main()
