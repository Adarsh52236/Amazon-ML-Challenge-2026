import os
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

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
    }

def main():
    print("Loading Phase 7 Buggy OOS Predictions (which is exactly Cap 100 Production Ranking)...")
    with open(ROOT / "output/phase7_oos_results.pkl", "rb") as f:
        results = pickle.load(f)
        
    preds = results['Buggy']['full_preds']
    
    # Reconstruct Fold Column
    folds = []
    seen = set()
    current_fold = 1
    for s in preds['source1_entity_id']:
        if s not in seen:
            if len(seen) == 1000 * current_fold:
                current_fold += 1
            seen.add(s)
        folds.append(current_fold)
    preds['fold'] = folds
    
    gt_mapping = load_ground_truth(as_dict=True)
    dev_gt = {s: gt_mapping.get(s, []) for s in preds['source1_entity_id'].unique()}
    
    thresholds = [x / 100 for x in range(60, 81)]
    
    results_global = []
    results_folds = []
    
    for t in thresholds:
        m = evaluate_fold_preds(preds, dev_gt, t)
        results_global.append({
            'Threshold': t,
            'Macro_F05': m['Macro_F05'],
            'Precision': m['Precision'],
            'Recall': m['Recall'],
            'ZM_FP_Rate': m['ZM_FP_Rate'],
            'Singleton_Acc': m['Singleton_Acc'],
            'MultiMatch_F05': m['MultiMatch_F05']
        })
        
        for f in range(1, 6):
            fm = evaluate_fold_preds(preds[preds['fold'] == f], dev_gt, t)
            results_folds.append({
                'Threshold': t,
                'Fold': f,
                'Macro_F05': fm['Macro_F05']
            })
            
    df_global = pd.DataFrame(results_global)
    df_folds = pd.DataFrame(results_folds)
    
    print("\n2. EVALUATE A FINE THRESHOLD GRID")
    print(df_global.to_string(index=False))
    
    print("\n3. FOLD-LEVEL STABILITY (0.65 - 0.78)")
    for t in [x/100 for x in range(65, 79)]:
        f_scores = df_folds[df_folds['Threshold'] == t]['Macro_F05']
        print(f"Threshold {t:.2f}: Mean={f_scores.mean():.4f}, Std={f_scores.std():.4f}, Min={f_scores.min():.4f}, Max={f_scores.max():.4f}")
        for i, val in enumerate(f_scores):
            print(f"  Fold {i+1}: {val:.4f}")
            
    print("\n4. IDENTIFY THE OOS PLATEAU")
    max_f05 = df_global['Macro_F05'].max()
    print(f"Maximum Macro F0.5: {max_f05:.4f}")
    
    within_0005 = df_global[df_global['Macro_F05'] >= max_f05 - 0.0005]
    within_0010 = df_global[df_global['Macro_F05'] >= max_f05 - 0.0010]
    
    print("\nThresholds within 0.0005 of max:")
    print(within_0005['Threshold'].tolist())
    
    print("\nThresholds within 0.0010 of max:")
    print(within_0010['Threshold'].tolist())
    
    print("\n6. PRECISION / RECALL TRADEOFF (0.68 - 0.76)")
    sub = df_global[(df_global['Threshold'] >= 0.68) & (df_global['Threshold'] <= 0.76)]
    print(sub[['Threshold', 'Macro_F05', 'Precision', 'Recall', 'ZM_FP_Rate', 'Singleton_Acc']].to_string(index=False))
    
if __name__ == "__main__":
    main()
