import os
import sys
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from business_entity_resolution.data_loader import load_ground_truth, load_training_data
from business_entity_resolution.candidate_generator import CandidateGenerator
from business_entity_resolution.preprocessing_config import BlockingConfig

def calculate_macro_f05(preds_dict, gt_dict, all_s1_ids):
    f_scores = []
    tp_total = fp_total = fn_total = 0
    
    # Track entity-level stats
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
        entity_stats[s1_id] = {
            'tp': tp, 'fp': fp, 'fn': fn, 'f05': f05,
            'gt_len': len(gt_set), 'pred_len': len(pred_set)
        }
        
    macro_f05 = np.mean(f_scores)
    
    # Global pairwise stats
    global_prec = tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else 0.0
    global_rec = tp_total / (tp_total + fn_total) if tp_total + fn_total > 0 else 0.0
    global_f1 = (2 * global_prec * global_rec) / (global_prec + global_rec) if global_prec + global_rec > 0 else 0.0
    
    return macro_f05, global_prec, global_rec, global_f1, entity_stats

def main():
    t0 = time.time()
    
    # 1. Load Data & Models
    feat_path = ROOT / "data/processed/features_dev_phase5_improved.parquet"
    df = pd.read_parquet(feat_path)
    
    gt_dict = load_ground_truth(as_dict=True)
    
    all_s1_ids = df['source1_entity_id'].unique().tolist()
    
    print("Loading models...")
    model_xgb = xgb.Booster()
    model_xgb.load_model(str(ROOT / "models/model_xgb_prod.json"))
    model_lgb = lgb.Booster(model_file=str(ROOT / "models/model_lgb_prod.txt"))
    
    feature_cols = [c for c in df.columns if c not in ["source1_entity_id", "target_entity_id", "target_source"]]
    X = df[feature_cols].values
    
    print("Predicting...")
    dmat = xgb.DMatrix(X, feature_names=feature_cols)
    p_xgb = model_xgb.predict(dmat)
    p_lgb = model_lgb.predict(X)
    p_blend = 0.75 * p_xgb + 0.25 * p_lgb
    
    # EXP 3 & 4: THRESHOLD SWEEPS
    def evaluate_sweep(probs, name):
        results = []
        for thresh in np.arange(0.05, 1.0, 0.05):
            preds_dict = defaultdict(list)
            mask = probs >= thresh
            for i, is_match in enumerate(mask):
                if is_match:
                    preds_dict[df['source1_entity_id'].values[i]].append(df['target_entity_id'].values[i])
            
            macro_f05, g_prec, g_rec, g_f1, estats = calculate_macro_f05(preds_dict, gt_dict, all_s1_ids)
            
            # Additional stats
            zm_entities = [s for s in all_s1_ids if estats[s]['gt_len'] == 0]
            zm_fp_rate = sum(1 for s in zm_entities if estats[s]['fp'] > 0) / len(zm_entities) if zm_entities else 0.0
            
            sg_entities = [s for s in all_s1_ids if estats[s]['gt_len'] == 1]
            sg_acc = sum(1 for s in sg_entities if estats[s]['tp'] == 1 and estats[s]['fp'] == 0) / len(sg_entities) if sg_entities else 0.0
            
            mm_entities = [s for s in all_s1_ids if estats[s]['gt_len'] > 1]
            mm_f05 = np.mean([estats[s]['f05'] for s in mm_entities]) if mm_entities else 0.0
            
            results.append({
                'Model': name,
                'Threshold': float(thresh),
                'Macro_F05': macro_f05,
                'Global_F1': g_f1,
                'Global_Prec': g_prec,
                'Global_Rec': g_rec,
                'ZM_FP_Rate': zm_fp_rate,
                'Singleton_Acc': sg_acc,
                'MultiMatch_F05': mm_f05
            })
        return pd.DataFrame(results), preds_dict, estats
        
    print("Sweeping XGB...")
    xgb_res, _, _ = evaluate_sweep(p_xgb, 'XGB')
    print("Sweeping LGB...")
    lgb_res, _, _ = evaluate_sweep(p_lgb, 'LGB')
    print("Sweeping Blend...")
    blend_res, blend_preds_dict, blend_estats = evaluate_sweep(p_blend, 'Blend')
    
    # Save sweep results
    sweep_df = pd.concat([xgb_res, lgb_res, blend_res])
    sweep_df.to_csv(ROOT / "output/macro_f05_threshold_sweep.tsv", sep="\t", index=False)
    
    # Evaluate breakdown at champion threshold (XGB @ 0.45, Blend @ 0.45)
    mask_champ = p_xgb >= 0.45
    preds_champ = defaultdict(list)
    for i, is_match in enumerate(mask_champ):
        if is_match:
            preds_champ[df['source1_entity_id'].values[i]].append(df['target_entity_id'].values[i])
            
    champ_macro_f05, _, _, _, champ_estats = calculate_macro_f05(preds_champ, gt_dict, all_s1_ids)
    
    # Dump breakdown JSON
    with open(ROOT / "output/champ_estats.json", "w") as f:
        json.dump(champ_estats, f, indent=2)

    # Output simple summary
    print(xgb_res.sort_values('Macro_F05', ascending=False).head(3))
    
    # Output arrays for zero-match / singleton analysis
    df['p_xgb'] = p_xgb
    df['p_blend'] = p_blend
    df.to_parquet(ROOT / "output/preds_df.parquet")
    
if __name__ == "__main__":
    main()
