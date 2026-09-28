import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

def load_gt():
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT / "src"))
    from business_entity_resolution.data_loader import load_ground_truth
    return load_ground_truth(as_dict=True)

def analyze_errors():
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    with open(ROOT / "output/phase7_oos_results.pkl", "rb") as f:
        res = pickle.load(f)['Buggy']
        
    preds = res['full_preds']
    gt_dict = load_gt()
    
    threshold = 0.75
    
    preds_dict = defaultdict(list)
    mask = preds['prob'] >= threshold
    for s1_id, t_id in zip(preds[mask]['source1_entity_id'], preds[mask]['target_entity_id']):
        preds_dict[s1_id].append(t_id)
        
    cand_dict = defaultdict(list)
    for s1_id, t_id in zip(preds['source1_entity_id'], preds['target_entity_id']):
        cand_dict[s1_id].append(t_id)
        
    all_s1 = preds['source1_entity_id'].unique()
    
    blocking_failures = 0
    ranking_failures = 0
    fn_count = 0
    fp_count = 0
    zm_fp_count = 0
    
    for s1_id in all_s1:
        gt_set = set(gt_dict.get(s1_id, []))
        pred_set = set(preds_dict.get(s1_id, []))
        cand_set = set(cand_dict.get(s1_id, []))
        
        # False negatives inside candidates (scored below threshold)
        fn_set = (gt_set & cand_set) - pred_set
        fn_count += len(fn_set)
        
        # False positives
        fp_set = pred_set - gt_set
        fp_count += len(fp_set)
        
        if len(gt_set) == 0 and len(fp_set) > 0:
            zm_fp_count += len(fp_set)
            
    print(f"Model FN (inside candidates): {fn_count}")
    print(f"Model FP: {fp_count}")
    print(f"Zero-Match FP: {zm_fp_count}")
    
if __name__ == '__main__':
    analyze_errors()
