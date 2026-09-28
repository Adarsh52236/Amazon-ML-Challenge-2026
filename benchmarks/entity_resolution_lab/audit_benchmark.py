import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import recordlinkage
from recordlinkage.datasets import load_febrl4

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "benchmarks" / "entity_resolution_lab"))

def main():
    print("Loading FEBRL4 dataset for audit...")
    dfA, dfB, true_links = load_febrl4(return_links=True)
    
    # Replicate Normalization
    for df in [dfA, dfB]:
        for col in ['given_name', 'surname', 'postcode', 'state', 'suburb', 'address_1', 'date_of_birth']:
            df[col] = df[col].astype(str).str.lower().str.strip()
            df[col] = df[col].replace('nan', np.nan)
            
    # Replicate Split
    np.random.seed(42)
    a_indices = dfA.index.values.copy()
    np.random.shuffle(a_indices)
    train_idx = a_indices[:3000]
    val_idx = a_indices[3000:4000]
    test_idx = a_indices[4000:]
    
    # Replicate Blocking
    indexer = recordlinkage.Index()
    indexer.block('given_name')
    indexer.block('surname')
    indexer.block('postcode')
    candidate_links = indexer.index(dfA, dfB)
    
    # Split candidates
    cand_df = pd.DataFrame(index=candidate_links)
    train_cands = cand_df[cand_df.index.get_level_values(0).isin(train_idx)].index
    val_cands = cand_df[cand_df.index.get_level_values(0).isin(val_idx)].index
    test_cands = cand_df[cand_df.index.get_level_values(0).isin(test_idx)].index
    
    true_set = set(true_links)
    
    # Accounting for TEST
    test_gt_total = sum(1 for (a,b) in true_links if a in test_idx)
    test_cand_set = set(test_cands)
    test_gt_in_cands = test_cand_set.intersection(true_set)
    test_gt_lost = test_gt_total - len(test_gt_in_cands)
    
    pos_cands_test = len(test_gt_in_cands)
    neg_cands_test = len(test_cands) - pos_cands_test
    
    # Analyze blocking errors in TEST
    blocking_errors = []
    for (a,b) in true_links:
        if a in test_idx and (a,b) not in test_cand_set:
            blocking_errors.append((a,b))
            
    err_reasons = {'given_name_mismatch': 0, 'surname_mismatch': 0, 'postcode_mismatch': 0, 'missing_value': 0, 'simultaneous_corruption': 0}
    for (a,b) in blocking_errors:
        recA = dfA.loc[a]
        recB = dfB.loc[b]
        
        gn_match = recA['given_name'] == recB['given_name']
        sn_match = recA['surname'] == recB['surname']
        pc_match = recA['postcode'] == recB['postcode']
        
        has_nan = pd.isna(recA['given_name']) or pd.isna(recB['given_name']) or \
                  pd.isna(recA['surname']) or pd.isna(recB['surname']) or \
                  pd.isna(recA['postcode']) or pd.isna(recB['postcode'])
                  
        if not gn_match: err_reasons['given_name_mismatch'] += 1
        if not sn_match: err_reasons['surname_mismatch'] += 1
        if not pc_match: err_reasons['postcode_mismatch'] += 1
        if has_nan: err_reasons['missing_value'] += 1
        if not gn_match and not sn_match and not pc_match:
            err_reasons['simultaneous_corruption'] += 1
            
    # Output to TSV
    audit_data = [{
        'Metric': 'test_gt_total', 'Value': test_gt_total
    }, {
        'Metric': 'test_gt_in_cands', 'Value': len(test_gt_in_cands)
    }, {
        'Metric': 'test_gt_lost_blocking', 'Value': test_gt_lost
    }, {
        'Metric': 'test_cands_total', 'Value': len(test_cands)
    }, {
        'Metric': 'test_cands_positive', 'Value': pos_cands_test
    }, {
        'Metric': 'test_cands_negative', 'Value': neg_cands_test
    }, {
        'Metric': 'train_cands_total', 'Value': len(train_cands)
    }, {
        'Metric': 'val_cands_total', 'Value': len(val_cands)
    }]
    
    for k, v in err_reasons.items():
        audit_data.append({'Metric': f'blocking_error_{k}', 'Value': v})
        
    out_dir = ROOT / "output/benchmarks"
    pd.DataFrame(audit_data).to_csv(out_dir / "febrl4_methodology_audit.tsv", sep="\t", index=False)
    
    print("Done generating audit data.")

if __name__ == "__main__":
    main()
