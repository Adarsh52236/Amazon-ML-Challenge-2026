import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
import recordlinkage
from recordlinkage.datasets import load_febrl4
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent.parent

def compute_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1, f05, tp, fp, fn

def main():
    start_time = time.time()
    print("Loading FEBRL4 dataset...")
    dfA, dfB, true_links = load_febrl4(return_links=True)
    
    # 1. Normalization
    for df in [dfA, dfB]:
        for col in ['given_name', 'surname', 'postcode', 'state', 'suburb', 'address_1', 'date_of_birth']:
            # Do not replace NaN with empty string for blocking keys because it causes massive blocks
            df[col] = df[col].astype(str).str.lower().str.strip()
            df[col] = df[col].replace('nan', np.nan)
    
    # Create deterministic splits on Table A
    np.random.seed(42)
    a_indices = dfA.index.values.copy()
    np.random.shuffle(a_indices)
    
    train_idx = a_indices[:3000]
    val_idx = a_indices[3000:4000]
    test_idx = a_indices[4000:]
    
    # 2. Candidate Generation (Blocking)
    print("Generating candidates...")
    indexer = recordlinkage.Index()
    # Block on given_name, surname, postcode
    indexer.block('given_name')
    indexer.block('surname')
    indexer.block('postcode')
    
    candidate_links = indexer.index(dfA, dfB)
    
    # Audit candidates
    true_set = set(true_links)
    cand_set = set(candidate_links)
    gt_lost = true_set - cand_set
    
    print(f"Total possible pairs: {len(dfA) * len(dfB)}")
    print(f"Candidate pairs generated: {len(candidate_links)}")
    print(f"Candidate reduction ratio: {len(dfA) * len(dfB) / len(candidate_links):.2f}")
    print(f"Ground-truth candidate recall: {len(true_set & cand_set) / len(true_set):.4f}")
    print(f"GT pairs lost during blocking: {len(gt_lost)}")
    
    # 3. Feature Extraction
    print("Extracting features...")
    compare = recordlinkage.Compare()
    compare.string('given_name', 'given_name', method='jarowinkler', label='given_name_sim')
    compare.string('surname', 'surname', method='jarowinkler', label='surname_sim')
    compare.exact('date_of_birth', 'date_of_birth', label='dob_match')
    compare.string('suburb', 'suburb', method='jarowinkler', label='suburb_sim')
    compare.string('state', 'state', method='jarowinkler', label='state_sim')
    compare.string('address_1', 'address_1', method='jarowinkler', label='address_sim')
    
    features = compare.compute(candidate_links, dfA, dfB)
    features['label'] = features.index.isin(true_links).astype(int)
    
    # Split data
    feat_train = features.loc[features.index.get_level_values(0).isin(train_idx)].copy()
    feat_val = features.loc[features.index.get_level_values(0).isin(val_idx)].copy()
    feat_test = features.loc[features.index.get_level_values(0).isin(test_idx)].copy()
    
    X_train = feat_train.drop('label', axis=1)
    y_train = feat_train['label']
    
    # 4. Train Model
    print("Training classifier...")
    model = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    model.fit(X_train, y_train)
    
    feat_val['prob'] = model.predict_proba(feat_val.drop('label', axis=1))[:, 1]
    feat_test['prob'] = model.predict_proba(feat_test.drop('label', axis=1))[:, 1]
    
    # 5. Threshold Analysis
    print("Running threshold analysis...")
    best_f05 = -1
    best_thresh = 0
    thresh_results = []
    
    for th in np.arange(0.1, 0.95, 0.05):
        y_pred = (feat_val['prob'] >= th).astype(int)
        p, r, f1, f05, tp, fp, fn = compute_metrics(feat_val['label'], y_pred)
        thresh_results.append({'threshold': round(th, 2), 'prec': p, 'rec': r, 'f1': f1, 'f05': f05})
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = th
            
    print(f"Best threshold on val: {best_thresh:.2f}")
    
    # Evaluate on test
    y_pred_test = (feat_test['prob'] >= best_thresh).astype(int)
    p, r, f1, f05, tp, fp, fn = compute_metrics(feat_test['label'], y_pred_test)
    test_res = {'prec': float(p), 'rec': float(r), 'f1': float(f1), 'f05': float(f05), 'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}
    
    # 6. Candidate Cap Ablation
    print("Running cap ablation on test set...")
    cap_results = []
    test_gt_total = sum(1 for (a,b) in true_links if a in test_idx)
    
    # Prepare cap ranking
    feat_test_sorted = feat_test.sort_values(by=['prob'], ascending=False)
    feat_test_sorted['rank'] = feat_test_sorted.groupby(level=0).cumcount() + 1
    
    for cap in [25, 50, 100, 200, 500, 99999]:
        cap_df = feat_test_sorted[feat_test_sorted['rank'] <= cap].copy()
        y_pred_cap = (cap_df['prob'] >= best_thresh).astype(int)
        
        c_p, c_r, c_f1, c_f05, c_tp, c_fp, c_fn = compute_metrics(cap_df['label'], y_pred_cap)
        
        # Recalculate true recall relative to test_gt_total
        final_recall = c_tp / test_gt_total if test_gt_total > 0 else 0
        cand_recall = len(cap_df[cap_df['label'] == 1]) / test_gt_total if test_gt_total > 0 else 0
        
        cap_results.append({
            'cap': 'Uncapped' if cap == 99999 else cap,
            'candidate_count': len(cap_df),
            'candidate_recall': cand_recall,
            'f05': c_f05,
            'f1': c_f1,
            'true_matches_lost': test_gt_total - c_tp
        })
        
    # 7. Error Analysis (False Positives and False Negatives)
    errors = []
    test_preds = feat_test_sorted[feat_test_sorted['rank'] <= 99999].copy()
    test_preds['pred'] = (test_preds['prob'] >= best_thresh).astype(int)
    
    fn_df = test_preds[(test_preds['label'] == 1) & (test_preds['pred'] == 0)]
    fp_df = test_preds[(test_preds['label'] == 0) & (test_preds['pred'] == 1)]
    
    if len(fn_df) > 0:
        a_id, b_id = fn_df.index[0]
        errors.append({'type': 'False Negative (Model)', 'a': dfA.loc[a_id].to_dict(), 'b': dfB.loc[b_id].to_dict(), 'prob': float(fn_df.iloc[0]['prob'])})
        
    if len(fp_df) > 0:
        a_id, b_id = fp_df.index[0]
        errors.append({'type': 'False Positive (Model)', 'a': dfA.loc[a_id].to_dict(), 'b': dfB.loc[b_id].to_dict(), 'prob': float(fp_df.iloc[0]['prob'])})
        
    # Blocking failures
    if len(gt_lost) > 0:
        a_id, b_id = list(gt_lost)[0]
        errors.append({'type': 'False Negative (Blocking)', 'a': dfA.loc[a_id].to_dict(), 'b': dfB.loc[b_id].to_dict(), 'prob': 0.0})
        
    # Output Results
    out_dir = ROOT / "output/benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(cap_results).to_csv(out_dir / "entity_resolution_results.tsv", sep="\t", index=False)
    
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            'blocking': {
                'total_pairs': len(dfA) * len(dfB),
                'candidate_pairs': len(candidate_links),
                'reduction_ratio': len(dfA) * len(dfB) / len(candidate_links),
                'gt_recall': len(true_set & cand_set) / len(true_set),
                'gt_lost': len(gt_lost)
            },
            'test_results': test_res,
            'threshold_sweep': thresh_results,
            'errors': errors
        }, f, indent=2)

if __name__ == "__main__":
    main()
